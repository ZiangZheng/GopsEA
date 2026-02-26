import torch
import statistics
from GopsEA import configclass
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from typing import TYPE_CHECKING, Literal
from GopsEA.utils.template.module_base import ModuleBaseCfg

if TYPE_CHECKING:
    from GopsEA.runners.on_policy.on_policy_runner import OnPolicyRunner


class LoggerBase:
    cfg: "LoggerBaseCfg"
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0

    # --------------------------------------------------------------------- #
    # initialization
    # --------------------------------------------------------------------- #
    def init_logger(self):
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.logger.lower()

            if self.logger_type == "neptune":
                from .utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

            elif self.logger_type == "wandb":
                from .utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

    def save_model(self, saved_dict, path, iter):
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, iter)

    def log_dict_infos(self, tab, it, tag, pad, writer):
        ret_string = ""
        for key, value in tab.items():
            if isinstance(value, (int, float)):
                pass
            elif isinstance(value, torch.Tensor) and value.numel() > 0:
                value = value.mean().item()
            else:
                continue
            writer.add_scalar(
                f"{tag}/{key}", value, it
            )
            ret_string += f"{f'{tag}/':>{self.cfg.tag_pad}}{f'{key}:':>{pad-self.cfg.tag_pad}} " + f"{value:.4f}\n"
        return ret_string
    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def log(
        self,
        runner: "OnPolicyRunner",
        locs: dict,
        width: int = None,
        pad: int = None,
        ep_string: str = ""
    ):
        width   = self.cfg.width if width is None else width
        pad     = self.cfg.pad if pad is None else pad
        # update global counters
        self._update_time_counters(runner, locs)

        log_string = ""
        log_string += self._log_header_string(locs, width)
        log_string += ep_string
        
        _string = self._log_episode_infos(runner, locs, pad)
        if self.cfg.is_log_ep_info : log_string += _string 
        _string = self._log_alg_update_infos(runner, locs, pad)
        if self.cfg.is_log_update  : log_string += _string 
        _string = self._log_sample_infos(runner, locs, pad)  
        if self.cfg.is_log_sample  : log_string += _string 
          
        log_string += self._log_statistics_string(runner, locs, pad)
        log_string += self._log_footer_string(runner, locs, pad, width)
        print(log_string)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _update_time_counters(self, runner, locs):
        steps = runner.cfg.num_steps_per_env * runner.env.num_envs
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_timesteps += steps
        self.tot_time += iteration_time

    # --------------------------- string ----------------------------------- #
    def _log_header_string(self, locs, width):
        title = f" Learning iteration {locs['it']}/{locs['tot_iter']} "
        return (
            f"{'#' * width}\n"
            f"{title.center(width, ' ')}\n"
        )

    def _log_episode_infos(self, runner, locs, pad):
        ep_string = ""
        if not locs["ep_infos"]: return ep_string
        for key in locs["ep_infos"][0]:
            values = []
            for ep_info in locs["ep_infos"]:
                if key not in ep_info: continue
                val = ep_info[key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor([val], device=runner.device)
                val = val.flatten()
                values.append(val)
            if not values: continue
            value = torch.cat(values).float().mean()
            if "/" in key:
                self.writer.add_scalar(key, value, locs["it"])
                ep_string += f"{f'{key}:':>{pad}}"+ f" {value:.4f}\n"
            else:
                self.writer.add_scalar(f"Episode/{key}", value, locs["it"])
                ep_string += f"{f'Mean episode {key}:':>{pad}}"+f" {value:.4f}\n"
        return ep_string

    def _log_alg_update_infos(self, runner, locs, pad):
        return self.log_dict_infos(tab=locs["alg_update_infos"], it=locs["it"], tag="Update", pad=pad, writer=self.writer)

    def _log_sample_infos(self, runner, locs, pad):
        return self.log_dict_infos(tab=locs["sample_infos"], it=locs["it"], tag="Sample", pad=pad, writer=self.writer)

    def _log_statistics_string(self, runner, locs, pad):
        if len(runner.rewbuffer) == 0:
            return ""
        mean_rew = statistics.mean(runner.rewbuffer)
        mean_len = statistics.mean(runner.lenbuffer)
        self.writer.add_scalar("Train/mean_reward", mean_rew, locs["it"])
        self.writer.add_scalar(
            "Train/mean_episode_length", mean_len, locs["it"]
        )

        if self.logger_type != "wandb":
            self.writer.add_scalar(
                "Train/mean_reward/time", mean_rew, self.tot_time
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time", mean_len, self.tot_time
            )

        return (
            f"{'Mean reward:':>{pad}} {mean_rew:.2f}\n"
            f"{'Mean episode length:':>{pad}} {mean_len:.2f}\n"
        )

    def _log_footer_string(self, runner, locs, pad, width):
        sample_infos = locs["sample_infos"]
        fps = int(runner.cfg.num_steps_per_env * runner.env.num_envs / (sample_infos["collection_time"] + locs["learn_time"]))
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        self.writer.add_scalar("Perf/collection_time", sample_infos["collection_time"], locs["it"])
        ret_string = (
            f"{'Computation:':>{pad}} " + f"{fps:.0f} steps/s \n"
        )
        iteration_time = sample_infos["collection_time"] + locs["learn_time"]
        eta = (
            self.tot_time
            / (locs["it"] + 1)
            * (locs["num_learning_iterations"] - locs["it"])
        )
        return ret_string + (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {eta:.1f}s\n"
        )

@configclass
class LoggerBaseCfg(ModuleBaseCfg):
    class_type: type[LoggerBase] = LoggerBase
    
    width           : int = 100
    pad             : int = 50
    tag_pad : int = 20
    
    logger          : Literal["tensorboard", "neptune", "wandb"] = "tensorboard"

    is_log_ep_info  : bool = False
    is_log_update   : bool = True
    is_log_sample   : bool = True

    neptune_project : str = "GopsEA" # """The neptune project name. Default is "GopsEA"."""
    wandb_project   : str = "GopsEA" # """The wandb project name. Default is "GopsEA"."""