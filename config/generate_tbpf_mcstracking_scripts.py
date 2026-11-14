#!/usr/bin/env python
"""
Generate (and submit) the jobs of a Tb+PF MCS tracking campaign over many months. The months are tracked at the same time, their track numbers
are stitched into one file, and the steps that need the whole tracks run over the stitched range, so a track or MCS that crosses the boundary of
two months stays one track (docs/stitching.md, docs/tbpf_mcs_workflow.md).

The months are split into chunks of --chunk-months months (default 12: one year is one chunk). For a chunk of one month the job is the plain job
of all steps. For a chunk of several months:

  track<YYYYMM>        one job per month, all submitted at once, one node each: idfeature, tracksingle, gettracks of that month. The end date
                       of a month is the start date of the next one, so neighbouring months share the cloudid file of that instant.
  stats<first>_<last>  one job per chunk, after all the track jobs of the chunk: first the links to the cloudid / track files of the months in
                       one folder (runscripts/link_period_folders.py) and the stitched track numbers (runscripts/stitch_tracknumbers.py),
                       then trackstats, identifymcs, matchpf, robustmcs, mapfeature and speed, and the optional rainmap and trackpcp over the
                       whole chunk. The stitching is a few minutes inside this job, not a job of its own (a job costs a further queue wait).

Every job starts only after the jobs it needs finished without error (sbatch --dependency=afterok, --kill-on-invalid-dep=yes: if a track job
fails, the stats job of its chunk is cancelled, not left pending). NOTHING IS SUBMITTED WITHOUT --submit: the files are written and the job
list is printed.

    python generate_tbpf_mcstracking_scripts.py -s 2009-01 -e 2009-12 -c config_mcs_tbpf_e3sm_0p1deg_template.yml -o campaign_2009 \\
        --input-dir /pscratch/.../olr_pcp --root-path /pscratch/.../mcs_tbpf/2009 --account m1657 [--stats-nodes 4] [--submit]

The config template has the placeholders STARTDATE, ENDDATE and YEAR-MONTH (the period and the folder of the month) and INPUT_DIR, TRACK_DIR
(--input-dir, --root-path); the template decides whether the optional steps rainmap and trackpcp run (run_rainmap, run_trackpcp), --rainmap and
--trackpcp override it. A job that failed is written again (the same command) and only what is needed is submitted; a job that is not
submitted in that call is taken as finished:

    python generate_tbpf_mcstracking_scripts.py ... --submit --jobs track200906 stats200901_200912

Author: Zhe Feng | zhe.feng@pnnl.gov
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CONFIG_DIR)
RUNSCRIPT_DIR = os.path.join(REPO, 'runscripts')
SLURM_TRACK_TEMPLATE = os.path.join(CONFIG_DIR, 'slurm_tbpf_mcstracking_track.sh')
SLURM_STATS_TEMPLATE = os.path.join(CONFIG_DIR, 'slurm_tbpf_mcstracking_stats_multinode.sh')
DEFAULT_RUN_SCRIPT = 'run_mcs_tbpf_unified.py'
DEFAULT_PYTHON_ENV = '/global/common/software/m1867/python/pyflex26.3'
# The steps of the jobs: the core steps of a track job and of a stats job, and the optional steps that the config template decides
CORE_TRACK_STEPS = ('idfeature', 'tracksingle', 'gettracks')
CORE_STATS_STEPS = ('trackstats', 'identifymcs', 'matchpf', 'robustmcs', 'mapfeature', 'speed')
OPTIONAL_TRACK_STEPS = ('preprocess',)
OPTIONAL_STATS_STEPS = ('rainmap', 'trackpcp', 'remap_healpix', 'regrid_mask')
# CPUs (hardware threads) of a Dask worker: one core with its hyperthread for a worker of one thread (a Perlmutter CPU node has 256 of them)
CPUS_PER_WORKER = 2


########################################################################################################################
# Periods and chunks
########################################################################################################################
def month_range(start, end):
    """
    The months of a campaign as periods.

    Args:
        start, end: string
            First and last month, YYYY-MM.

    Returns:
        months: list of dictionaries
            In time order: tag (YYYYMM), name (YYYY-MM, the folder of the month), start and end (YYYYMMDD.HHMM: 00:00 of the first day of the
            month and of the next month, so the end date of a month is the start date of the next one).

    Raises:
        SystemExit: the last month is before the first.
    """
    first, last = datetime.strptime(start, '%Y-%m'), datetime.strptime(end, '%Y-%m')
    if last < first:
        raise SystemExit(f"ERROR: the last month {end} is before the first month {start}")
    months, current = [], first
    while current <= last:
        # The first day of the next month
        following = current.replace(year=current.year + (current.month == 12), month=1 if current.month == 12 else current.month + 1)
        months.append({'tag': current.strftime('%Y%m'), 'name': current.strftime('%Y-%m'),
                       'start': current.strftime('%Y%m%d.0000'), 'end': following.strftime('%Y%m%d.0000')})
        current = following
    return months


def chunks_of(periods, size=None):
    """
    Split the periods into the groups that are tracked separately and stitched inside the group.

    Args:
        periods: list
            Output of month_range.
        size: int (default: None)
            Periods per group; None: one group with all of them.

    Returns:
        groups: list of lists of periods
            The last group has the periods that are left.

    Raises:
        SystemExit: size is less than 1.
    """
    if size is not None and size < 1:
        raise SystemExit(f"ERROR: --chunk-months must be 1 or more, not {size}")
    size = len(periods) if size is None else size
    return [periods[i:i + size] for i in range(0, len(periods), size)]


########################################################################################################################
# Config text
########################################################################################################################
def format_value(value):
    """
    A Python value as it is written in a YAML config line.

    Args:
        value: bool, int, float, string or None
            The value.

    Returns:
        text: string
            True / False, null, the number, or the string (quoted if it has characters that YAML reads otherwise).
    """
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, (int, float)):
        return str(value)
    return str(value) if re.fullmatch(r'[A-Za-z0-9_./~+-]+', str(value)) else "'" + str(value).replace("'", "''") + "'"


def set_key(text, key, value):
    """
    Set a top-level 'key: value' line of a config text, keeping its comment; the key is appended when the text has none.

    Args:
        text: string
            Config text.
        key: string
            Key name.
        value: bool, int, float, string or None
            New value (format_value).

    Returns:
        text: string

    Raises:
        SystemExit: the key is in the text more than once.
    """
    pattern = re.compile(rf'^({re.escape(key)}\s*:\s*)[^#\n]*?(\s*(?:#.*)?)$', re.M)
    found = pattern.findall(text)
    if len(found) > 1:
        raise SystemExit(f"ERROR: the config text has the key '{key}' {len(found)} times")
    if not found:
        return text.rstrip('\n') + f"\n{key}: {format_value(value)}\n"
    return pattern.sub(lambda m: f"{m.group(1)}{format_value(value)}{m.group(2) if m.group(2).strip() else ''}", text, count=1)


def template_flag(text, step):
    """
    The run_<step> flag of a config text: False if the key is not there.

    Args:
        text: string
            Config text (placeholders are fine, they are strings).
        step: string
            Step name without run_.

    Returns:
        flag: bool
    """
    return bool((yaml.safe_load(text) or {}).get(f'run_{step}', False))


def run_flags(kind, text, overrides=None):
    """
    The run_<step> flags of a job.

    A track job runs the core track steps (and preprocess if the template has it on), a stats job the core stats steps and the optional
    steps that the template has on, the plain job of one month all the core steps and the optional steps of the template. The options
    --rainmap and --trackpcp replace the flag of the template for rainmap and trackpcp.

    Args:
        kind: string
            'track', 'stats' or 'month'.
        text: string
            Config text of the template.
        overrides: dictionary (default: None)
            {step: bool} that replaces the template's flag of an optional step.

    Returns:
        flags: dictionary
            {step: bool} for every step that a job can run.
    """
    overrides = overrides or {}
    flag = lambda step: overrides.get(step, template_flag(text, step))
    on_track, on_stats = kind in ('track', 'month'), kind in ('stats', 'month')
    flags = {s: on_track for s in CORE_TRACK_STEPS}
    flags.update({s: on_stats for s in CORE_STATS_STEPS})
    flags.update({s: on_track and flag(s) for s in OPTIONAL_TRACK_STEPS})
    flags.update({s: on_stats and flag(s) for s in OPTIONAL_STATS_STEPS})
    return flags


def fill_config(text, period, input_dir, root_path, flags, overrides=None):
    """
    The config of one job: the placeholders filled, the run flags set and the keys of the job.

    Args:
        text: string
            Config template text.
        period: dictionary
            The period of the job (start, end, name).
        input_dir, root_path: string or None
            Replace INPUT_DIR / TRACK_DIR (with a trailing slash); a template without the placeholder is left as it is.
        flags: dictionary
            {step: bool}, see run_flags.
        overrides: dictionary (default: None)
            {key: value} to set (e.g. tracking_path_name, run_parallel).

    Returns:
        text: string
    """
    text = text.replace('STARTDATE', period['start']).replace('ENDDATE', period['end']).replace('YEAR-MONTH', period['name'])
    if input_dir is not None:
        text = text.replace('INPUT_DIR', input_dir.rstrip('/'))
    if root_path is not None:
        text = text.replace('TRACK_DIR', root_path.rstrip('/'))
    for step, value in flags.items():
        # A flag that the template does not have and that is off is not added
        if value or re.search(rf'^run_{re.escape(step)}\s*:', text, re.M):
            text = set_key(text, f'run_{step}', value)
    for key, value in (overrides or {}).items():
        text = set_key(text, key, value)
    return text


def folders_of(config_text):
    """
    Folders and file name base of a config text without placeholders, as load_config builds them.

    Args:
        config_text: string
            Config text.

    Returns:
        folders: dictionary
            root, tracking_name (tracking_path_name), tracking (cloudid and track files) and stats (tracknumbers files), both with a trailing
            slash, and tracknumbers_filebase.
    """
    c = yaml.safe_load(config_text)
    return {'root': c['root_path'], 'tracking_name': c['tracking_path_name'],
            'tracking': os.path.join(c['root_path'], c['tracking_path_name']).rstrip('/') + '/',
            'stats': os.path.join(c['root_path'], c['stats_path_name']).rstrip('/') + '/',
            'tracknumbers_filebase': c.get('tracknumbers_filebase', 'tracknumbers_')}


def stitch_lines(group, month_folders, link_folder, stats_folder, filebase):
    """
    The shell lines of a stats job that join the periods of its chunk: the folder of links, then the stitched tracknumbers file.

    Args:
        group: list of dictionaries
            The periods of the chunk.
        month_folders: list of strings
            Tracking folder of every period, in the order of group.
        link_folder: string
            Folder to fill with links to the cloudid and track files of all the periods.
        stats_folder: string
            Folder of the tracknumbers files (with trailing slash).
        filebase: string
            tracknumbers_filebase of the config.

    Returns:
        lines: string
            Shell lines; each stops the job (exit 1) if it fails.
    """
    q = shlex.quote
    # The tracknumbers file of a range of periods: the name gettracks gives it (filebase + startdate + _ + enddate)
    name = lambda first, last: f"{stats_folder}{filebase}{first['start']}_{last['end']}.nc"
    link = ['python', os.path.join(RUNSCRIPT_DIR, 'link_period_folders.py'), link_folder] + month_folders
    stitch = (['python', os.path.join(RUNSCRIPT_DIR, 'stitch_tracknumbers.py'), name(group[0], group[-1])]
              + [name(m, m) for m in group] + ['--tracking-dirs'] + month_folders)
    return ("# Join the periods of this chunk: one folder with links to their files, the stitched tracknumbers file\n"
            f"{' '.join(q(a) for a in link)} || exit 1\n"
            f"{' '.join(q(a) for a in stitch)} || exit 1\n")


########################################################################################################################
# Checks of the template
########################################################################################################################
def check_template(text):
    """
    Check a config template before any file is written.

    Args:
        text: string
            Config template text.

    Raises:
        SystemExit: a placeholder of the period is missing in startdate, enddate or tracking_path_name, the text is not a YAML mapping, or the robust MCS step is on and the robust MCS
        definition (robustmcs_definition) or the keys of its definition are not there.
    """
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise SystemExit("ERROR: the config template is not a YAML mapping")
    # The period and the folder of the month are placeholders in the values of these keys (a comment that names them does not count)
    for key, token in (('startdate', 'STARTDATE'), ('enddate', 'ENDDATE'), ('tracking_path_name', 'YEAR-MONTH')):
        if token not in str(config.get(key, '')):
            raise SystemExit(f"ERROR: the config template needs the placeholder {token} in {key}")
    # The robust MCS definition is required, with its keys, when the step is on (the run script stops at its start otherwise)
    if config.get('run_robustmcs', False):
        sys.path.insert(0, REPO)
        from pyflextrkr.robustmcs_rows import robustmcs_params
        try:
            robustmcs_params(config)
        except ValueError as err:
            raise SystemExit(f"ERROR: {err}")


def warn_if_folder_used(root, tracking_name):
    """
    Print a warning if the tracking folder of a period already has files: the jobs overwrite files, and a stats job links every file of a folder.

    Args:
        root: string
            root_path of the outputs.
        tracking_name: string
            tracking_path_name (relative to root).
    """
    folder = os.path.join(root, tracking_name)
    if os.path.isdir(folder) and os.listdir(folder):
        print(f"WARNING: {folder} already has files; the jobs overwrite them. Use a root that holds only this campaign.")


########################################################################################################################
# Slurm scripts
########################################################################################################################
def slurm_script(template_text, tokens):
    """
    A slurm script from a template: the @TOKEN@ placeholders replaced.

    Args:
        template_text: string
            Slurm template text.
        tokens: dictionary
            {TOKEN: value} (without the @).

    Returns:
        text: string

    Raises:
        SystemExit: a placeholder is left.
    """
    text = template_text
    for token, value in tokens.items():
        text = text.replace(f'@{token}@', str(value))
    left = sorted(set(re.findall(r'@[A-Z_]+@', text)))
    if left:
        raise SystemExit(f"ERROR: the slurm template has placeholders that were not filled: {', '.join(left)}")
    return text


def dependency_option(job_ids):
    """
    The sbatch options that make a job wait for jobs to finish without error.

    Args:
        job_ids: list of strings
            Job ids.

    Returns:
        options: list of strings
            ['--dependency=afterok:1:2', '--kill-on-invalid-dep=yes']; [] for no jobs.
    """
    if not job_ids:
        return []
    return ['--dependency=afterok:' + ':'.join(job_ids), '--kill-on-invalid-dep=yes']


########################################################################################################################
# The campaign
########################################################################################################################
def build_campaign(periods, config_template, output_dir, input_dir=None, root_path=None, run_script=DEFAULT_RUN_SCRIPT, chunk_size=None,
                   account='', qos='regular', python_env=DEFAULT_PYTHON_ENV, mail_user=None, track_time='03:00:00', stats_time='06:00:00',
                   stats_nodes=1, workers_per_node=64, worker_memory='7GB', overrides=None):
    """
    Write the configs and the slurm scripts of a campaign and return its jobs.

    Args:
        periods: list of dictionaries
            The periods (month_range): tag, name, start, end; neighbouring periods share the file of the instant that is the end of one and the
            start of the next.
        config_template: string
            Config template file.
        output_dir: string
            Folder for the files (created, with logs/).
        input_dir, root_path: string (default: None)
            Replace INPUT_DIR and TRACK_DIR of the template.
        run_script: string (default: 'run_mcs_tbpf_unified.py')
            A path, or a name in the runscripts folder.
        chunk_size: int (default: None)
            Periods per chunk; None: one chunk with all of them.
        account, qos: string (default: '', 'regular')
            Slurm account and QOS.
        python_env: string (default: DEFAULT_PYTHON_ENV)
            Conda environment of the jobs.
        mail_user: string (default: None)
            E-mail address for the job mails; None: no mail lines.
        track_time, stats_time: string (default: '03:00:00', '06:00:00')
            Wallclock of the track / month jobs and of the stats jobs (sbatch --time syntax).
        stats_nodes: int (default: 1)
            Nodes of a stats job; more than 1 uses the multi-node template (a Dask scheduler and workers_per_node workers on every node) and
            run_parallel: 2 in its config.
        workers_per_node, worker_memory: int, string (default: 64, '7GB')
            Dask workers per node and their memory limit (multi-node stats job).
        overrides: dictionary (default: None)
            {step: bool} for the optional steps rainmap and trackpcp (replacing the flags of the template).

    Returns:
        jobs: list of dictionaries
            In submission order: name, kind ('month': all steps of one period, 'track', 'stats'), script (path) and needs (names of the jobs it
            waits for).
    """
    with open(config_template) as f:
        template = f.read()
    check_template(template)
    run_script_path = run_script if os.path.isfile(run_script) else os.path.join(RUNSCRIPT_DIR, run_script)
    if not os.path.isfile(run_script_path):
        raise SystemExit(f"ERROR: run script not found: {run_script} (also looked in {RUNSCRIPT_DIR})")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    with open(SLURM_TRACK_TEMPLATE) as f:
        track_template = f.read()
    with open(SLURM_STATS_TEMPLATE) as f:
        stats_template = f.read()
    config_base = os.path.splitext(os.path.basename(config_template))[0]
    mail_lines = f"#SBATCH --mail-user={mail_user}\n#SBATCH --mail-type=END,FAIL" if mail_user else ''
    jobs = []

    def add_job(name, kind, config_text, needs=(), prelude='', wallclock=track_time, multinode=False):
        """Write the config and the slurm script of a job and add it to jobs."""
        config_path = os.path.join(output_dir, f"{config_base}_{name}.yml")
        with open(config_path, 'w') as f:
            f.write(config_text)
        tokens = {'ACCOUNT': account, 'QOS': qos, 'JOB_NAME': name, 'WALLCLOCK': wallclock, 'MAIL_LINES': mail_lines,
                  'PYTHON_ENV': python_env, 'PRELUDE': prelude, 'RUN_SCRIPT': run_script_path, 'CONFIG': config_path,
                  'NODES': stats_nodes, 'WORKERS_PER_NODE': workers_per_node, 'WORKER_MEMORY': worker_memory,
                  'CPUS_PER_WORKER': CPUS_PER_WORKER}
        script_path = os.path.join(output_dir, f"slurm_{name}.sh")
        with open(script_path, 'w') as f:
            f.write(slurm_script(stats_template if multinode else track_template, tokens))
        os.chmod(script_path, 0o755)
        jobs.append({'name': name, 'kind': kind, 'script': script_path, 'needs': list(needs), 'config': config_path})

    for group in chunks_of(periods, chunk_size):
        first, last = group[0], group[-1]
        # A chunk of one period is the plain job of all steps
        if len(group) == 1:
            flags = run_flags('month', template, overrides)
            text = fill_config(template, first, input_dir, root_path, flags)
            folders = folders_of(text)
            warn_if_folder_used(folders['root'], folders['tracking_name'])
            add_job(f"month{first['tag']}", 'month', text)
            continue
        # Track jobs: idfeature, tracksingle and gettracks of every period, all at once
        for period in group:
            text = fill_config(template, period, input_dir, root_path, run_flags('track', template, overrides))
            folders = folders_of(text)
            warn_if_folder_used(folders['root'], folders['tracking_name'])
            add_job(f"track{period['tag']}", 'track', text)
        # Stats job: link and stitch the periods, then the other steps over the whole chunk; its tracking folder (the links) is a sibling of
        # the monthly folders
        tag = f"{first['tag']}_{last['tag']}"
        month_folders = [folders_of(fill_config(template, p, input_dir, root_path, {}))['tracking'] for p in group]
        base = folders_of(fill_config(template, first, input_dir, root_path, {}))
        link_name = os.path.join(os.path.dirname(base['tracking_name'].rstrip('/')), f'campaign_{tag}')
        prelude = stitch_lines(group, month_folders, os.path.join(base['root'], link_name), base['stats'], base['tracknumbers_filebase'])
        multi = stats_nodes > 1
        stats_overrides = {'tracking_path_name': link_name}
        if multi:
            stats_overrides['run_parallel'] = 2
        text = fill_config(template, dict(first, end=last['end']), input_dir, root_path, run_flags('stats', template, overrides), stats_overrides)
        add_job(f'stats{tag}', 'stats', text, needs=[f"track{p['tag']}" for p in group], prelude=prelude, wallclock=stats_time, multinode=multi)
    with open(os.path.join(output_dir, 'jobs.json'), 'w') as f:
        json.dump(jobs, f, indent=1)
    return jobs


def submit(jobs, output_dir, names=None, sbatch='sbatch'):
    """
    Submit jobs in order, each waiting for the jobs it needs.

    Args:
        jobs: list of dictionaries
            Output of build_campaign.
        output_dir: string
            Folder the jobs are submitted from (the relative logs/ path of #SBATCH --output).
        names: list of strings (default: None)
            Submit only these jobs; a job they need that is not in the list is taken as finished. None: all jobs.
        sbatch: string (default: 'sbatch')
            The submit command.

    Returns:
        ids: dictionary
            {job name: Slurm job id} of the submitted jobs.

    Raises:
        SystemExit: a name in names is not a job, or the submission of a job failed.
    """
    unknown = sorted(set(names or []) - {j['name'] for j in jobs})
    if unknown:
        raise SystemExit(f"ERROR: no such job: {', '.join(unknown)} (jobs: {', '.join(j['name'] for j in jobs)})")
    ids = {}
    # Loop over the jobs in submission order: the ids of the jobs it needs are known by then
    for job in jobs:
        if names is not None and job['name'] not in names:
            continue
        waits = [n for n in job['needs'] if n in ids]
        cmd = [sbatch, '--parsable'] + dependency_option([ids[n] for n in waits]) + [job['script']]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=output_dir)
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"ERROR: submitting {job['name']} failed: {e.stderr.strip()}\njobs submitted before it: {ids if ids else 'none'}")
        ids[job['name']] = result.stdout.strip().split(';')[0]
        print(f"  submitted {job['name']:24s} job {ids[job['name']]}" + (f"  after {', '.join(waits)}" if waits else ''))
    return ids


def main():
    """Command line: write the files of a campaign and, with --submit, submit them."""
    parser = argparse.ArgumentParser(description='Generate (and submit) a Tb+PF MCS tracking campaign: months tracked at once, stitched, then '
                                                 'the steps over whole tracks. Nothing is submitted without --submit.')
    parser.add_argument('-s', '--start', required=True, help='first month, YYYY-MM')
    parser.add_argument('-e', '--end', required=True, help='last month, YYYY-MM')
    parser.add_argument('-c', '--config-template', required=True, help='config template with the placeholders STARTDATE, ENDDATE, YEAR-MONTH')
    parser.add_argument('-o', '--output-dir', required=True, help='folder for the configs and scripts')
    parser.add_argument('--input-dir', default=None, help='replaces INPUT_DIR of the template (the input data)')
    parser.add_argument('--root-path', default=None,
                        help='replaces TRACK_DIR of the template (root_path of the outputs): one folder per campaign, never one that already holds '
                             'another run of the same months')
    parser.add_argument('-r', '--run-script', default=DEFAULT_RUN_SCRIPT, help='run script: a path, or a name in runscripts/ (default: %(default)s)')
    parser.add_argument('--chunk-months', type=int, default=12,
                        help='months per chunk, stitched inside a chunk (default: %(default)s: a year; 1: plain monthly jobs)')
    parser.add_argument('--account', required=True, help='Slurm account')
    parser.add_argument('--qos', default='regular', help='Slurm QOS (default: %(default)s)')
    parser.add_argument('--python-env', default=DEFAULT_PYTHON_ENV, help='conda environment of the jobs (default: %(default)s)')
    parser.add_argument('--mail-user', default=None, help='e-mail address for the job mails (default: none)')
    parser.add_argument('--track-time', default='03:00:00', help='wallclock of the track jobs, sbatch --time syntax (default: %(default)s)')
    parser.add_argument('--stats-time', default='06:00:00', help='wallclock of the stats jobs, sbatch --time syntax (default: %(default)s)')
    parser.add_argument('--stats-nodes', type=int, default=1, help='nodes of a stats job; more than 1 uses the multi-node template (default: %(default)s)')
    parser.add_argument('--workers-per-node', type=int, default=64, help='Dask workers per node of a multi-node stats job (default: %(default)s)')
    parser.add_argument('--worker-memory', default='7GB', help='memory limit of a Dask worker (default: %(default)s)')
    parser.add_argument('--rainmap', choices=['on', 'off'], default=None, help='the monthly rain maps: on or off (default: run_rainmap of the template)')
    parser.add_argument('--trackpcp', choices=['on', 'off'], default=None,
                        help='the precipitation accumulated along the tracks: on or off (default: run_trackpcp of the template)')
    parser.add_argument('--submit', action='store_true', help='submit the jobs (default: only write the files)')
    parser.add_argument('--jobs', nargs='+', default=None, help='with --submit: only these jobs (names are printed); the jobs '
                        'they need that are not listed are taken as finished')
    args = parser.parse_args()
    if not os.path.isfile(args.config_template):
        parser.error(f'file not found: {args.config_template}')
    overrides = {step: value == 'on' for step, value in (('rainmap', args.rainmap), ('trackpcp', args.trackpcp)) if value is not None}
    jobs = build_campaign(month_range(args.start, args.end), args.config_template, args.output_dir, args.input_dir, args.root_path, args.run_script,
                          args.chunk_months, args.account, args.qos, args.python_env, args.mail_user, args.track_time, args.stats_time,
                          args.stats_nodes, args.workers_per_node, args.worker_memory, overrides)
    print(f"{len(jobs)} jobs, files in {os.path.abspath(args.output_dir)}:")
    for job in jobs:
        print(f"  {job['name']:24s} {job['kind']:6s} " + (f"after {', '.join(job['needs'])}" if job['needs'] else 'at once'))
    if args.submit:
        print('Submitting:')
        submit(jobs, os.path.abspath(args.output_dir), args.jobs)
    else:
        print('Nothing submitted (add --submit).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
