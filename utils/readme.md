# Model Zoo Port Utilities

Utility files for making the Modelzoo port from `monolith`

## List of files and brief usage

- [export.sh](./export.sh) - Used to export the code state in master branch to [modelzoo](https://github.com/Cerebras/modelzoo) repo.
- [header.txt](./header.txt) - License that gets added to all the code and configs.
- [files-no-header.txt](./files-no-header.txt) - List of files in modelzoo-internal for which the license won't be added.
- [mz-files-exclude.txt](./mz-files-exclude.txt) - list of files that are in modelzoo, abd should not be ported from monolith.
- [src-files-exclude.txt](./src-files-exclude.txt) - list of files that are in monolith that we want to skip. Only the files in the sub dir of the ported dirs.
- [file-mapping.txt](./file-mapping.txt) - Used to remap the file path. Mainly used in customer repo. **Do not add any modelzoo paths here or else config and readme porting will break.**
- [readme_dirs.txt](./readme_dirs.txt) - Used to list all the model paths which should have the readme to be ported from monolith.
- [config_dirs.txt](./config_dirs.txt) - Used to list all the model paths which should have the yaml files to be ported from monolith.
- -[vocab_file_mapping.txt](./vocab_file_mapping.txt) - Vocab file names are mangled from `src/models` to `modelzoo/transformers/vocab`, this file lists that mapping.

### To port readme and configs

Generate a list of mapping from `src/models` to `modelzoo` folder by running

```python
python utils/make_modelzoo.py -s ~/ws/monolith/src/ -p modelzoo/ -m readme_and_configs
```
