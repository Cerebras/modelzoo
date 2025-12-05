#!/usr/bin/env python3
"""
make_modelzoo.py - Script for porting models from monolith to modelzoo.

Owner: ryan@cerebras.net
"""

import argparse
import logging
import os
import pathlib
import re
import shutil
import subprocess

MZ_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(MZ_UTILS_DIR, 'header.txt')
MZ_FILES_EXCLUDE = os.path.join(MZ_UTILS_DIR, 'mz-files-exclude.txt')
SRC_FILES_EXCLUDE = os.path.join(MZ_UTILS_DIR, 'src-files-exclude.txt')
FILES_NO_HEADER = os.path.join(MZ_UTILS_DIR, 'files-no-header.txt')
FILE_MAPPING = os.path.join(MZ_UTILS_DIR, 'file-mapping.txt')
README_DIRS = os.path.join(MZ_UTILS_DIR, "readme_dirs.txt")
CONFIG_DIRS = os.path.join(MZ_UTILS_DIR, "config_dirs.txt")
VOCAB_MAPPING = os.path.join(MZ_UTILS_DIR, "vocab_file_mapping.txt")

BEGIN_CEREBRAS_ONLY = re.compile(r'[.#\s]*BEGIN_CEREBRAS_ONLY')
END_CEREBRAS_ONLY = re.compile(r'[#\s]*END_CEREBRAS_ONLY')
IMPORT = re.compile(r'(\s*)import\s+(customer\.models|modelzoo)\.([^\s]+)(.*)')
IMPORT_RELATIVE = re.compile(r'(\s*import\s\.(.+)|\s*from\s\.(.+))')
IMPORT_FROM = re.compile(
    r'(\s*)from\s+(customer\.models|modelzoo)\.([\w\.]+)\s+import\s+(.+)'
)
MZ_CODE = re.compile(r'(\s*)#\s*MZ:\s{0,1}(.*)')
ISORT = re.compile(r'(\s*)#\s*isort:\s*(.*)')
BEGIN_MZ_DEDENT = re.compile(r'[.#\s]*BEGIN_MZ_DEDENT')
END_MZ_DEDENT = re.compile(r'[.#\s]*END_MZ_DEDENT')


logging.basicConfig(level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--src',
        default=None,
        help='Path of monolith/src directory (required).',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='./',
        help='Path of output directory (default is ./).',
    )
    parser.add_argument(
        '-p',
        '--prefix',
        default="",
        help="Only files that start with --prefix (wrt --output) will be ported.",
    )
    parser.add_argument(
        '-a',
        '--add_files',
        default=False,
        action='store_true',
        help='Add files from src that are not already in modelzoo.',
    )
    parser.add_argument(
        '-i',
        '--resolve_imports',
        default=False,
        action='store_true',
        help='Attempt to resolve imports to files not already in modelzoo.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=False,
        action='store_true',
        help='Toggle verbose output.',
    )
    parser.add_argument(
        '-d',
        '--dryrun',
        default=False,
        action='store_true',
        help='Just walkthrough and print files; make no edits.',
    )
    parser.add_argument(
        '-m',
        "--mode",
        choices=["code", "readme", "configs", "scripts",],
        default="code",
        help="Can be either code (.py files), readme (.md files), configs "
             "(.yaml files) or scripts (limited to src/user_scripts).",
    )
    return parser.parse_args()


def get_gittop():
    """
    https://github.com/Cerebras/monolith/blob/master/src/common/python/git.py#L24
    """
    # try to determine it by running git commands:
    cmd = ["git", "rev-parse", "--show-toplevel"]
    try:
        gittop = subprocess.check_output(
            cmd, universal_newlines=True, cwd="."
        ).strip()
    except subprocess.CalledProcessError as ex:
        logging.exception(' '.join(cmd) + ": " + ex.output)
        logging.error(ex.output)

    gittop = _run_git_cmd_get_out(cmd, cwd=".")
    if gittop and os.path.isdir(gittop):
        return gittop
    # looks like we're not in a git repo:
    return None


def get_file_list(file_list_path):
    """
    Get list of files from file_list_path, which has one file per line.
    """
    file_list = []
    for line in open(file_list_path, 'r'):
        line = line.strip()
        if line:
            # normpath needed to ignore trailing `/`
            file_list.append(os.path.normpath(line))
    return file_list


def get_git_files(dir_path, extensions=None, prefix=""):
    """
    Get git-tracked files starting from dir_path w/ given extensions, prefix.
    """
    cwd = os.path.join(dir_path, prefix)
    files = subprocess.check_output(["git", "ls-files"], cwd=cwd).split()
    files = [os.path.join(prefix, path.decode()) for path in files]
    if extensions is not None:
        files = [
            path
            for path in files
            if any([path.endswith(ext) for ext in extensions])
        ]
    return set(files)


class MZUpdater:
    def __init__(self, args):
        self.src_dir = os.path.realpath(args.src)
        self.mz_dir = args.output
        self.prefix = args.prefix
        self.add_files = args.add_files
        self.resolve_imports = args.resolve_imports
        self.verbose = args.verbose
        self.dryrun = args.dryrun
        self.mode = args.mode

        if self.mode == "code":
            self.extensions = [".py"]
            self.set_env_vars()  # for cspyformatter
        elif self.mode == "scripts":
            self.extensions = None  # all the files
        elif self.mode == "readme":
            self.extensions = [".md", ".png", "jpg", ".jpeg"]
        elif self.mode == "configs":
            self.extensions = [".yaml"]
        else: 
            logging.info("Unsupported mode passed. Supported modes are ")
        # TODO: update the mz_dir so non-updated files aren't listed on console

        self.existing_files = get_git_files(
            self.mz_dir, extensions=self.extensions
        )
        src_prefix = "models"
        if self.mode == "readme":
            # Setup a list of files from monolith to be ported
            # return the init and in the updater, just copy readme
            # files
            self.src_files = get_git_files(
                self.src_dir, extensions=self.extensions, prefix=src_prefix
            )
            self.update_src_for_readme()
        elif self.mode == "configs":
            # Setup a list of files from monolith to be ported
            # return the init and in the updater, just copy readme
            # files
            self.src_files = get_git_files(
                self.src_dir, extensions=self.extensions, prefix=src_prefix
            )
            self.update_src_for_configs()
        else:
            if self.mode == "scripts":
                src_prefix = "user_scripts"
            self.src_files = get_git_files(
                self.src_dir, extensions=self.extensions, prefix=src_prefix
            )
            if self.mode == "code":
                self.src_files.update(
                    get_git_files(
                        self.src_dir,
                        extensions=self.extensions,
                        prefix="customer_models",
                    )
                )

            self.checked_imports = set()
            self.format_errored_file_list = []

        self.mz_exclude_list = get_file_list(MZ_FILES_EXCLUDE)
        self.src_exclude_list = get_file_list(SRC_FILES_EXCLUDE)

        self.unported_mz_files = {
            path for path in self.existing_files if self.should_update(path)
        }

        # files to not append header to
        self.no_header_list = get_file_list(FILES_NO_HEADER)

        # create file/import mappings
        self.file_mapping = []
        self.import_mapping = []
        for line in open(FILE_MAPPING, 'r'):
            line = line.strip()
            if line:
                from_path, to_path = line.split(":")
                from_path = from_path.strip()
                to_path = to_path.strip()
                self.file_mapping.append([from_path, to_path])
                from_import = from_path.replace("/", ".")
                to_import = to_path.replace("/", ".")
                self.import_mapping.append([from_import, to_import])

    def set_env_vars(self):
        """
        Update env variables for cspyformatter to run
        """
        monolith_dir = os.path.dirname(self.src_dir)
        # cspyformatter resides in monolith $GITTOP/obj/bin
        # monolith needs to be built for this to be available
        cspyformatter_root = os.path.join(monolith_dir, "obj", "bin")
        assert (
            os.path.exists(cspyformatter_root)
        ), f"Please build monolith at {monolith_dir} for cspyformatter"
        
        # cspyformatter is a shell script which calls three utilities:
        # 1. autoflake
        # 2. black
        # 3. isort
        # all three needs to be referred from monolith and are soft-linked
        # at $GITTOP/python/python-x86_64/bin
        cspyformatter_dep_root = os.path.join(
            monolith_dir, "python", "python-x86_64", "bin"
        )
        assert os.path.exists(
            cspyformatter_dep_root
        ), f"Dependency path for cspyformatter not found, something is wrong"
        # add both the paths above to env var
        os.environ["PATH"] = ":".join(
            [cspyformatter_root, cspyformatter_dep_root, os.getenv("PATH")]
        )


    def update_src_for_readme(self):
        """
        Updates the list with all the files to be ported from monolith.
        Checks the global list of paths and removes all other ones. 
        """

        needed_readme_dirs = get_file_list(README_DIRS)
        remove_src_file = set()
        # loop in the current src files to only check for needed dirs
        for file_path in self.src_files:
            tgt_file_dir = os.path.dirname(file_path).replace("models", "modelzoo", 1)
            if tgt_file_dir not in needed_readme_dirs:
                remove_src_file.add(file_path)
        
        # update the src file list
        self.src_files.difference_update(remove_src_file)

    def update_src_for_configs(self):
        """
        Updates the list with all the files to be ported from monolith.
        Checks the global list of paths and removes all other ones. 
        """

        needed_config_dirs = get_file_list(CONFIG_DIRS)
        remove_src_file = set()
        # loop in the current src files to only check for needed dirs
        for file_path in self.src_files:
            tgt_file_dir = os.path.dirname(file_path)#.replace("models", "modelzoo", 1)
            if tgt_file_dir not in needed_config_dirs:
                remove_src_file.add(file_path)
        
        # update the src file list
        self.src_files.difference_update(remove_src_file)

        # store vocab file mapping
        self.vocab_file_mapping = {}
        for line in open(VOCAB_MAPPING, 'r'):
            line = line.strip()
            if line:
                model_name, vocab_path = line.split(":")
                self.vocab_file_mapping[model_name.strip()] = vocab_path.strip()

    def update_modelzoo(self):
        """
        Updates the modelzoo (self.mz_dir) using monolith/src (self.src_dir).
        """
        # map each mz_path that we need to update to corresponding src_path
        self.files_to_update = {}
        for src_path in self.src_files:
            mz_path = self.create_mz_path(src_path)
            if self.should_update(mz_path, src_path, verbose=self.verbose):
                self.files_to_update[mz_path] = src_path
                if mz_path in self.unported_mz_files:
                    self.unported_mz_files.remove(mz_path)

        if len(self.unported_mz_files) > 0:
            logging.info("")
            logging.warning("The following MZ files will not be updated:")
            for path in sorted(list(self.unported_mz_files)):
                logging.info(path)
            logging.info("")

        added_files = []

        while len(self.files_to_update) > 0:
            mz_path, src_path = self.files_to_update.popitem()
            if self.dryrun or self.verbose:
                logging.info(f"{src_path} --> {mz_path}")
            if not self.dryrun:
                if self.mode == "code":
                    self.update_code_file(src_path, mz_path)
                    self.run_cspyformatter(mz_path)
                elif self.mode == "scripts":
                    self.copy_files(src_path, mz_path)
                elif self.mode == "readme":
                    src_ext = os.path.splitext(src_path)[-1]
                    if src_ext == ".md":
                        self.update_readme_config_file(src_path, mz_path)
                    else:
                        self.copy_file(src_path, mz_path)
                elif self.mode == "configs":
                    self.update_readme_config_file(src_path, mz_path)
                else:
                    logging.error(
                        f"Unsupported case for porting the code. "
                        f"Received: {self.mode}")
                # check if the file was added
                if mz_path not in self.existing_files:
                    self.existing_files.add(mz_path)
                    added_files.append(mz_path)

        if self.mode == "code":
            if len(self.format_errored_file_list) > 0:
                logging.error(
                    "cspyformatter errored out on following files, please check "
                    "for any syntax errors in these files:"
                )
                for file_path in self.format_errored_file_list:
                    logging.info(file_path)

        if len(added_files) > 0:
            logging.info("New files were added in monolith!")
            for mz_path in added_files:
                logging.info(mz_path)
                while True:
                    resp = input("Would you like to git add this file (y/n)? ")
                    if resp.lower() == "y":
                        for mz_path in added_files:
                            p = subprocess.Popen(
                                ["git", "add", mz_path], cwd=self.mz_dir
                            )
                            p.wait()
                        break
                    elif resp.lower() == "n":
                        break

    def apply_file_mapping(self, path, reverse=False):
        """
        Checks if path starts with any keys in self.file_mapping. If so,
        replaces the start of that path with the corresponding value.

        If reverse is True, then the key and value in file_mapping are swapped.
        """
        mapping = reversed(self.file_mapping) if reverse else self.file_mapping
        for from_path, to_path in mapping:
            if reverse:
                from_path, to_path = to_path, from_path
            if path.startswith(from_path):
                path = path.replace(from_path, to_path, 1)
        return path

    def create_mz_path(self, src_path):
        """
        Given the path to a file relative to self.src_dir, generates the path
        to the corresponding location in self.mz_dir that the file should be
        ported to.
        """
        # generic path updates
        if src_path.startswith("models"):
            mz_path = src_path.replace("models", "modelzoo", 1)
        elif src_path.startswith("customer_models/"):
            mz_path = src_path.replace("customer_models/", "", 1)
        elif src_path.startswith("user_scripts"):
            mz_path = os.path.join("modelzoo", src_path)
        else:
            raise ValueError(f"Invalid source path: {src_path}")

        mz_path = self.apply_file_mapping(mz_path)

        return mz_path

    def should_update(self, mz_path, src_path=None, verbose=False):
        """
        Checks if mz_path should be updated by this script.
        """
        if not mz_path.startswith(self.prefix):
            return False
        if not self.add_files and mz_path not in self.existing_files:
            if verbose:
                logging.info('Not adding new file: %s' % (mz_path))
            return False
        if mz_path in self.mz_exclude_list:
            if verbose:
                logging.info('Skipping in MZ exclude_list: %s' % (mz_path))
            return False
        if src_path is not None and src_path in self.src_exclude_list:
            if verbose:
                logging.info('Skipping in src exclude_list: %s' % (src_path))
            return False
        return True

    def copy_file(self, src_path, mz_path):
        """
        Copy files from src_path to mz_path.
        """
        p_in = os.path.join(self.src_dir, src_path)
        p_out = os.path.join(self.mz_dir, mz_path)
        d_out = os.path.dirname(p_out)

        if not os.path.isdir(d_out):
            os.makedirs(d_out)
        try:  # rm p_out if it exists
            os.remove(p_out)
        except OSError:
            pass

        shutil.copy(p_in, p_out)

    def update_readme_config_file(self, src_path, mz_path):
        """
        Copy README files from monolith to modelzoo.
        - For images, it just copies the files.
        - For .md files, it strips off the content within CEREBRAS_ONLY tags.
        """
        p_in = os.path.join(self.src_dir, src_path)
        p_out = os.path.join(self.mz_dir, mz_path)
        d_out = os.path.dirname(p_out)

        if not os.path.isdir(d_out):
            os.makedirs(d_out)
        try:  # rm p_out if it exists
            os.remove(p_out)
        except OSError:
            pass


        model_dir = pathlib.Path(mz_path).parents[1]
        model_name = os.path.basename(model_dir)
        if model_name == "bert" and "roberta" in mz_path:
            model_name = "roberta"

        f_out = open(p_out, 'w')
        f_in = open(p_in, 'r')
        inside_ignore_block = False
        for line in f_in:
            # check for CEREBRAS_ONLY blocks
            if inside_ignore_block:
                if END_CEREBRAS_ONLY.match(line):
                    inside_ignore_block = False
                continue
            elif BEGIN_CEREBRAS_ONLY.match(line):
                inside_ignore_block = True
                continue
            assert not inside_ignore_block
            # modelzoo-specific content
            line = self.process_mz_code(line)
            if self.mode == "configs":
                line = self.process_config_line(line, model_name)
            # write line
            f_out.write(line)
        f_in.close()
        f_out.close()
        assert not inside_ignore_block, (
            f"{p_out} ended while inside a BEGIN_CEREBRAS_ONLY block."
        )

    def process_config_line(self, line, model_name):
        """
        Copy config files from monolith to modelzoo.
        - Removes Daria's username from the path
        - Removes `/cb/ml` or `/cb/datasets` from input path
        - TODO: Updates vocab filepath to relative instead of absolute
        """
        # replace the data dir path to hide /cb/
        if line.find("/cb/ml"):
            line = line.replace("/cb/ml/", "./", 1)
        elif line.find("/cb/datasets/"):
            line = line.replace("/cb/datasets/", "./", 1)
        # special case, remove daria's name
        line = line.replace("_daria", "", 1)
        # TODO: Update vocab path
        if re.search(r"\bvocab_file\b", line) and model_name in self.vocab_file_mapping.keys():
            line_beg = line.split(": ")[0]
            vocab_file = "".join([' "', self.vocab_file_mapping[model_name], '"\n'])
            line = ":".join([line_beg, vocab_file])
        return line

    def update_code_file(self, src_path, mz_path):
        """
        Copy file src_path and write a cleaned version to mz_path.

        -   Makes output directories if needed.
        -   Removes pre-existing output files.
        -   Adds header.txt to the top of *.py files.
        -   Ignores CEREBRAS_ONLY blocks of text/code.
        -   Adds MZ specific code.
        -   Detents code if it is in MZ_DEDENT block.
        -   Changes cerebras imports to modelzoo-style.
        -   Checks to make sure imports are valid after porting.

        Args:
            src_path (str): Path to copy from monolith relative to self.src_dir.
            mz_path (str): Path to write to in modelzoo relative to self.mz_dir.
        """
        p_in = os.path.join(self.src_dir, src_path)
        p_out = os.path.join(self.mz_dir, mz_path)
        d_out = os.path.dirname(p_out)

        if not os.path.isdir(d_out):
            os.makedirs(d_out)
        try:  # rm p_out if it exists
            os.remove(p_out)
        except OSError:
            pass

        f_out = open(p_out, 'w')

        if (
            mz_path not in self.no_header_list
            and os.path.dirname(mz_path) not in self.no_header_list
            and os.stat(p_in).st_size
        ):
            f_h = open(HEADER, 'r')
            f_out.writelines(f_h.readlines())
            f_out.write('\n')
            f_h.close()
        elif self.verbose:
            logging.info('Not adding header to %s' % (mz_path))

        f_in = open(p_in, 'r')
        inside_ignore_block = False
        inside_dedent_block = False
        for line in f_in:
            # skip isort for sys.path.append fixes
            if ISORT.match(line):
                continue
            # check for CEREBRAS_ONLY blocks
            if inside_ignore_block:
                if END_CEREBRAS_ONLY.match(line):
                    inside_ignore_block = False
                continue
            elif BEGIN_CEREBRAS_ONLY.match(line):
                inside_ignore_block = True
                continue
            assert not inside_ignore_block
            # check for MZ_DEDENT blocks and dedent
            if inside_dedent_block:
                if END_MZ_DEDENT.match(line):
                    inside_dedent_block = False
                    continue
                line = self.process_mz_ded_code(line, inside_dedent_block)
            elif BEGIN_MZ_DEDENT.match(line):
                inside_dedent_block = True
                continue
            # modelzoo-specific code
            line = self.process_mz_code(line)
            # change imports
            line = self.process_import(line, mz_path)
            # write line
            f_out.write(line)
        f_in.close()
        f_out.close()
        if inside_ignore_block:
            logging.info(
                'WARNING: %s ended while inside a BEGIN_CEREBRAS_ONLY block.'
                % (p_out)
            )

    def process_mz_code(self, line):
        """
        Removes `# MZ:` tag from line if present.
        """
        reo = MZ_CODE.match(line)
        if reo:
            line = "%s%s\n" % (reo.group(1), reo.group(2))
        return line

    def process_mz_ded_code(self, line, inside_dedent_block):
        """
        Detents the line.
        """
        assert inside_dedent_block
        reo = re.match('(\s*)(.*)', line)
        if reo:
            #TODO: hacky way right now, fix if any better solution 
            dedent_str = reo.group(1)[:-4]
            line = "%s%s\n" % (dedent_str, reo.group(2))
        return line

    def process_import(self, line, mz_path):
        """
        Updates import for modelzoo. Checks whether imported file is in MZ,
        and if appropriate, adds that file to self.files_to_update.
        """
        reo_import = IMPORT.match(line)
        reo_import_relative = IMPORT_RELATIVE.match(line)
        reo_import_from = IMPORT_FROM.match(line)
        if not (reo_import or reo_import_from or reo_import_relative):
            return line
        if reo_import: reo = reo_import 
        elif reo_import_from: reo = reo_import_from
        else: reo = reo_import_relative
        
        # taking mz_path and appending the import to the path
        if reo_import_relative:
            mz_import = (mz_path.rsplit('/',1)[0]+reo.string.split(" ")[1])
        else:
            # generate mz version of import
            if reo.group(2) == "modelzoo":
                mz_import = "modelzoo." + reo.group(3)
            else:
                mz_import = reo.group(3)
            for import_from, import_to in self.import_mapping:
                mz_import = mz_import.replace(import_from, import_to, 1)

            # update line
            if reo_import:
                line = '%simport %s%s\n' % (reo.group(1), mz_import, reo.group(4))
            else:
                line = '%sfrom %s import %s\n' % (
                    reo.group(1),
                    mz_import,
                    reo.group(4),
                )
            
        # if import is to a file that is (A) not in modelzoo already, and (B)
        # not in self.files_to_update, then we try to resolve that import by
        # porting an extra file from Monolith.
        mz_import_path = mz_import.replace(".", "/")
        if os.path.isdir(os.path.join(self.mz_dir, mz_import_path)):
            mz_import_path = os.path.join(mz_import_path, "__init__")
        mz_import_path += ".py"
        if (
            mz_import_path not in self.checked_imports
            and mz_import_path not in self.existing_files
            and mz_import_path not in self.files_to_update
        ):
            self.checked_imports.add(mz_import_path)
            logging.info(
                f"{mz_path} imports from {mz_import_path}, "
                f"which has not been ported to the modelzoo.",
            )
            if reo_import_relative:
                src_import_path = mz_import_path.replace("modelzoo", "models")
            else:
                src_import_path = (
                    "models." if reo.group(2) == "modelzoo" else "customer_models."
                )
                src_import_path += reo.group(3)
                src_import_path = src_import_path.replace(".", "/")
            # assert mz_import_path == self.create_mz_path(src_import_path)
            if os.path.isdir(os.path.join(self.src_dir, src_import_path)):
                src_import_path = os.path.join(src_import_path, "__init__")
            if not reo_import_relative:  src_import_path += ".py"     

            if src_import_path in self.src_files:
                if self.resolve_imports:
                    logging.info(
                        "Porting the following file from monolith/src"
                        "to resolve the import:"
                    )
                    self.files_to_update[mz_import_path] = src_import_path
                else:
                    logging.info(
                        "If this script is re-run with the -i flag,",
                        "the following file would be ported from monolith/src"
                        "to resolve the import:"
                    )
                logging.info(src_import_path)
            else:
                logging.warning(
                    "WARNING: Could not resolve import,",
                    f"since {src_import_path} is not in monolith/src.",
                )
            logging.info("")
        return line

    def run_cspyformatter(self, path):
        """
        Run cspyformatter.

        Args:
            path (str): Path of file to clean.
        """
        cmd = [f"cspyformatter {path}"]
        output = subprocess.run(
            cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        returncode = output.returncode
        assert returncode != 127, "cspyformatter not found! Is monolith built?"
        assert returncode != 126, "cspyformatter could not be run."
        assert (
            returncode != 2
        ), "Incorrect usage of cspyformatter. Input file is: {path}."
        if not returncode:  # successful execution
            if self.verbose:
                stderror = output.stderr
                if stderror: 
                    logging.info(stderror.decode("utf-8"))
                stdout = output.stdout
                if stdout: 
                    logging.info(stdout.decode("utf-8"))
            return
        # here because returncode is not 0, 2, 126 or 127
        if self.verbose:
            logging.info("cspyformatter returned unknown error:")
            logging.info(output.stderr.decode('utf-8'))
        self.format_errored_file_list.append(path)


def main():
    logging.getLogger().setLevel(logging.INFO)

    args = get_args()
    updater = MZUpdater(args)
    updater.update_modelzoo()
    
    # if args.mode == "code" or args.mode == "scripts":
    #     updater.update_modelzoo_code()
    # elif args.mode == "readme":
    #     updater.update_readme_files()
    # elif args.mode == "configs":
    #     updater.update_config_files()
    # else:
    #     logging.error(
    #         f"Unsupported case for porting the code. Received: {args.mode}"
    #     )


if __name__ == '__main__':
    main()
