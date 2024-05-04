import argparse
import copy
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.multiprocessing import spawn
from torch.optim.adam import Adam

from envs import *
from models import *
from utils import *
from videos import *

logger = get_logger(__file__)
writer = get_writer()


def train(cfg, verbose=False):
    if verbose:
        writer.save_hyper_params(cfg)

    env = create_env(cfg, verbose)

    shared_agent = ActorCritic(
        in_dim=cfg.agent_in_dim,
        hidden_dim=cfg.agent_hidden_dim,
        backbone=cfg.agent_net,
        temperature=cfg.agent_temperature,
    ).cuda()

    optim = Adam(shared_agent.parameters())
    baselines = [None for _ in range(cfg.node_cnt)]

    if cfg.resume_directory != "" and cfg.resume_checkpoint > 0:
        shared_agent_params = torch.load(f"{cfg.resume_directory}/agent_epi_{cfg.resume_checkpoint}.pt")
        optim_params = torch.load(f"{cfg.resume_directory}/optim_epi_{cfg.resume_checkpoint}.pt")
        baselines = torch.load(f"{cfg.resume_directory}/base_epi_{cfg.resume_checkpoint}.pt")

        shared_agent.load_state_dict(shared_agent_params)
        optim.load_state_dict(optim_params)

    for epi in range(cfg.resume_checkpoint + 1, cfg.episodes):
        sync_dist_model(shared_agent)
        sync_cuda_rng_state()

        agents = [copy.deepcopy(shared_agent).cpu() for _ in range(cfg.node_cnt)]

        with tc.collect_time("time/act"):
            states = env.reset()
            actions, log_probs = [], []

            for agent, n_state in zip(agents, states):
                agent.cuda(), agent.eval()
                n_action, n_log_prob = [], []

                for state in n_state:
                    state = state.cuda()
                    probs = agent.pi(state).view(-1, 2)

                    m = Categorical(probs)
                    action = m.sample()
                    log_prob = m.log_prob(action)

                    n_action.append(action.cpu().detach())
                    n_log_prob.append(log_prob.cpu().detach())

                actions.append(n_action)
                log_probs.append(n_log_prob)

        with tc.collect_time("time/step"):
            reactions = env.step(actions, log_probs)

        optim.zero_grad()

        for nid, agent, n_state, n_action, n_log_prob, n_reward in zip(range(cfg.node_cnt), agents, states, actions, log_probs, reactions["rewards"]):
            with tc.collect_time("time/train"):
                agent.cuda(), agent.train()
                if baselines[nid] is None:
                    baselines[nid] = torch.stack(n_reward).cuda()
                baselines[nid] = 0.9 * baselines[nid].cuda() + 0.1 * torch.stack(n_reward).cuda()

                n_loss = []
                for state, action, reward, baseline in zip(n_state, n_action, n_reward, baselines[nid]):
                    state = state.cuda().detach()
                    action = action.cuda().detach()
                    reward = reward.cuda().detach()
                    baseline = baseline.cuda().detach()

                    prob = agent.pi(state).view(-1, 2)
                    m = Categorical(prob)
                    log_prob = m.log_prob(action)

                    actor_loss = -(log_prob * (reward - baseline)).mean()
                    critic_loss = F.mse_loss(agent.v(state), baseline)

                    loss = cfg.lambda_actor_loss * actor_loss + cfg.lambda_critic_loss * critic_loss + cfg.lambda_reg_loss * ((log_prob - 0.5) ** 2).mean()

                    n_loss.append(loss)

                n_loss = sum(n_loss) / len(n_loss)
                n_loss.backward()

                # aggregate gradients
                for param, shared_param in zip(agent.parameters(), shared_agent.parameters()):
                    weight = len([len(a) for a in n_action]) / sum([len(a) for n_action in actions for a in n_action])
                    if shared_param.grad is not None:
                        shared_param._grad += weight * param.grad
                    else:
                        shared_param._grad = weight * param.grad

            if verbose:
                metrics = {
                    f"agent_{nid}_loss": n_loss.item(),
                    f"agent_{nid}_baseline": torch.mean(baselines[nid]).item(),
                    f"agent_{nid}_probs/mean": torch.cat([log_prob.exp() for log_prob in n_log_prob]).mean().item(),
                    f"agent_{nid}_probs/var": torch.cat([log_prob.exp() for log_prob in n_log_prob]).var().item(),
                    f"agent_{nid}_picks/sel_count": torch.cat(n_action).sum().item(),
                    f"agent_{nid}_picks/sel_ratio": torch.cat(n_action).float().mean().item(),
                    f"agent_{nid}_reward": np.mean(reactions["rewards"][nid]).mean(),
                    f"agent_{nid}_reward/div": np.mean(reactions["div_rewards"][nid]),
                    f"agent_{nid}_reward/rep": np.mean(reactions["rep_rewards"][nid]),
                    f"agent_{nid}_reward/eff": np.mean(reactions["eff_rewards"][nid]),
                    f"agent_{nid}_reward/acc": np.mean(reactions["acc_rewards"][nid]),
                    f"agent_{nid}_time/act": tc.times["time/act"],
                    f"agent_{nid}_time/step": tc.times["time/step"],
                    f"agent_{nid}_time/train": tc.times["time/train"],
                }

                if cfg.env_task == "classification":
                    metrics.update(
                        {
                            f"agent_{nid}_acc/top_1": reactions["acc/top_1"],
                            f"agent_{nid}_acc/top_3": reactions["acc/top_3"],
                            f"agent_{nid}_acc/top_5": reactions["acc/top_5"],
                        }
                    )
                elif cfg.env_task == "detection":
                    metrics.update(
                        {
                            f"agent_{nid}_acc/map": reactions["map"],
                            f"agent_{nid}_acc/map_50": reactions["map_50"],
                            f"agent_{nid}_acc/map_75": reactions["map_75"],
                            f"agent_{nid}_acc/map_small": reactions["map_small"],
                            f"agent_{nid}_acc/map_medium": reactions["map_medium"],
                            f"agent_{nid}_acc/map_large": reactions["map_large"],
                            f"agent_{nid}_acc/mar_1": reactions["mar_1"],
                            f"agent_{nid}_acc/mar_10": reactions["mar_10"],
                            f"agent_{nid}_acc/mar_100": reactions["mar_100"],
                            f"agent_{nid}_acc/mar_small": reactions["mar_small"],
                            f"agent_{nid}_acc/mar_medium": reactions["mar_medium"],
                            f"agent_{nid}_acc/mar_large": reactions["mar_large"],
                        }
                    )

                logger.info(f"Episode [{epi}/{cfg.episodes}]", extra={"metrics": metrics})
                writer.save_metrics(metrics, epi)

                if cfg.checkpoint_interval > 0 and epi % cfg.checkpoint_interval == 0:
                    torch.save(shared_agent.state_dict(), f"{writer.log_dir}/agent_epi_{epi}.pt")
                    torch.save(optim.state_dict(), f"{writer.log_dir}/optim_epi_{epi}.pt")
                    torch.save(baselines, f"{writer.log_dir}/base_epi_{epi}.pt")

        optim.step()


def create_env(args, verbose=False):
    # construct dataset
    if args.dataset == "core50_ni_c10":
        tr_data, val_data = core50_ni_c10(args.source_root)
    elif args.dataset == "core50_ni_c50":
        tr_data, val_data = core50_ni_c50(args.source_root)
    elif args.dataset == "core50_simulation":
        tr_data, val_data = core50_simulation(args.source_root)
    elif args.dataset == "visdrone_n1":
        tr_data, val_data = visdrone_n1(args.source_root)
    elif args.dataset == "visdrone_n3":
        tr_data, val_data = visdrone_n3(args.source_root)
    elif args.dataset == "visdrone_n5":
        tr_data, val_data = visdrone_n5(args.source_root)
    elif args.dataset == "visdrone_n8":
        tr_data, val_data = visdrone_n8(args.source_root)
    elif args.dataset == "mot15_n1":
        tr_data, val_data = mot15_n1(args.source_root)
    elif args.dataset == "mot15_n3":
        tr_data, val_data = mot15_n3(args.source_root)
    elif args.dataset == "mot15_n5":
        tr_data, val_data = mot15_n5(args.source_root)
    elif args.dataset == "mot15_n8":
        tr_data, val_data = mot15_n8(args.source_root)
    else:
        raise NotImplementedError

    # construct environment and model
    if args.model in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        classifier = ResNet(args.model, args.num_classes)
    elif args.model in ("vgg11", "vgg13", "vgg16", "vgg19"):
        classifier = VGG(args.model, args.num_classes)
    elif args.model in ("vit_tiny", "vit_small", "vit_base", "vit_large"):
        classifier = ViT(args.model, args.num_classes)
    elif args.model in ("swin_tiny", "swin_small", "swin_base", "swin_large"):
        classifier = ViT(args.model, args.num_classes)
    elif args.model == "cnn":
        classifier = CNN(args.num_classes)
    elif args.model in (
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
    ):
        detector = FRCNN(args.model, args.num_classes)
    else:
        raise NotImplementedError

    # construct environment
    if args.env_task == "classification":
        env = ClassificationTrainingEnv(
            node_cnt=args.node_cnt,
            window_size=args.window_size,
            classifier=classifier,
            tr_data=tr_data,
            val_data=val_data,
            compressed_rep_dim=args.compressed_rep_dim,
            lambda_div_reward=args.lambda_div_reward,
            lambda_rep_reward=args.lambda_rep_reward,
            lambda_eff_reward=args.lambda_eff_reward,
            lambda_acc_reward=args.lambda_acc_reward,
            verbose=verbose,
        )
    elif args.env_task == "detection":
        env = DetectionTrainingEnv(
            node_cnt=args.node_cnt,
            window_size=args.window_size,
            detector=detector,
            tr_data=tr_data,
            val_data=val_data,
            compressed_rep_dim=args.compressed_rep_dim,
            lambda_div_reward=args.lambda_div_reward,
            lambda_rep_reward=args.lambda_rep_reward,
            lambda_eff_reward=args.lambda_eff_reward,
            lambda_acc_reward=args.lambda_acc_reward,
            verbose=verbose,
        )
    else:
        raise NotImplementedError

    return env


def launch(rank, world_size, cfg):
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{cfg.hostname}:{cfg.port}",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)

    same_seeds(cfg.seed)
    train(cfg=cfg, verbose=(rank == 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset configuration
    parser.add_argument("--dataset", type=str, required=False, default="visdrone_n3")
    parser.add_argument("--source_root", type=str, required=False, default="/mnt/data/VisDrone2019-MOT")

    # video model configuration
    parser.add_argument("--model", type=str, required=False, default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--num_classes", type=int, required=False, default=12)

    # reinforced agent configuration
    parser.add_argument("--agent_net", type=str, required=False, default="transformer")
    parser.add_argument("--agent_in_dim", type=int, required=False, default=32)
    parser.add_argument("--agent_hidden_dim", type=int, required=False, default=32)
    parser.add_argument("--agent_temperature", type=float, required=False, default=1.0)

    # reinforced env configuration
    parser.add_argument("--episodes", type=int, required=False, default=100000)
    parser.add_argument("--env_task", type=str, required=False, default="detection")
    parser.add_argument("--node_cnt", type=int, required=False, default=3)
    parser.add_argument("--window_size", type=int, required=False, default=100000)
    parser.add_argument("--compressed_rep_dim", type=int, required=False, default=32)
    parser.add_argument("--lambda_div_reward", type=float, required=False, default=0.02)
    parser.add_argument("--lambda_rep_reward", type=float, required=False, default=0.03)
    parser.add_argument("--lambda_eff_reward", type=float, required=False, default=0.50)
    parser.add_argument("--lambda_acc_reward", type=float, required=False, default=1.00)
    parser.add_argument("--lambda_reg_loss", type=float, required=False, default=0.0001)
    parser.add_argument("--lambda_actor_loss", type=float, required=False, default=1.00)
    parser.add_argument("--lambda_critic_loss", type=float, required=False, default=0.10)

    # checkpoint and resume configuration
    parser.add_argument("--checkpoint_interval", type=int, required=False, default=100)
    parser.add_argument("--resume_checkpoint", type=int, required=False, default=0)
    parser.add_argument("--resume_directory", type=str, required=False, default="")

    # system configuration
    parser.add_argument("--hostname", type=str, required=False, default="127.0.0.1", help="Manual hostname")
    parser.add_argument("--port", type=int, required=False, default=8889, help="Manual port")
    parser.add_argument("--seed", type=int, required=False, default=42069, help="Manual seed")
    parser.add_argument("--device", type=str, required=False, default="", help="Manual GPU devices")
    parser.add_argument("--parallel", type=int, required=False, default=-1, help="Parallel size")

    cfg = parser.parse_args()

    if cfg.device != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device

    world_size = cfg.parallel
    if world_size == -1:
        world_size = torch.cuda.device_count()
    assert world_size >= 1, "No GPU found for training, please check by `torch.cuda.device_count()`."

    if world_size > 1:
        spawn(launch, args=(world_size, cfg), nprocs=world_size)
    else:
        launch(0, 1, cfg)
