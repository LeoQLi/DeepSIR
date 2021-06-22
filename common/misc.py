"""
Misc utilities
"""
import os
import sys
import shutil
import logging
import argparse
import git
import subprocess
import coloredlogs
from datetime import datetime

_logger = logging.getLogger()


def print_info(opt, log_dir=None):
    """ Logs source code configuration
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Print commit ID
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        _logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        if log_dir is not None:
            with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
                subprocess.run(['git', 'diff'], stdout=fid)

    except git.exc.InvalidGitRepositoryError:
        pass

    # Arguments
    arg_str = '\n'.join(['    {}: {}'.format(op, getattr(opt, op)) for op in vars(opt)])
    _logger.info('Arguments:\n' + arg_str)


def prepare_logger(opt: argparse.Namespace, log_path: str = None):
    """Creates logging directory, and installs colorlogs

    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace

    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory
    """

    if log_path is None:
        if opt.dev:
            log_path = '../logdev'
            shutil.rmtree(log_path, ignore_errors=True)
        else:
            datetime_str = datetime.now().strftime("%y%m%d_%H%M%S")
            if opt.name is not None:
                log_path = os.path.join(opt.logdir, datetime_str + '_' + opt.name)
            else:
                log_path = os.path.join(opt.logdir, datetime_str)

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s: %(message)s')

    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler('{}/log.txt'.format(log_path), mode='a')
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    print_info(opt, log_path)
    logger.info('Output and logs will be saved to: {}'.format(log_path))

    return logger, log_path
