#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

import sys
import importlib
import importlib.util
from pathlib import Path

package_name = 'nasbench_asr'

version_file = Path(__file__).parent.joinpath(package_name, 'version.py')
spec = importlib.util.spec_from_file_location('{}.version'.format(package_name), version_file)
package_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_version)
sys.modules[spec.name] = package_version


class build_maybe_inplace(build_py):
    def run(self):
        global package_version
        package_version = importlib.reload(package_version)
        _dist_file = version_file.parent.joinpath('_dist_info.py')
        assert not _dist_file.exists()
        _dist_file.write_text('\n'.join(map(lambda attr_name: attr_name+' = '+repr(getattr(package_version, attr_name)), package_version.__all__)) + '\n')
        return super().run()


setup(name='NasbenchASR',
      version=package_version.version,
      description='Library for the NasbenchASR dataset',
      author='SAIC-Cambridge, On-Device Team',
      author_email='on.device@samsung.com',
      url='https://github.sec.samsung.net/a-mehrotra1/pytorch-asr',
      download_url='https://github.sec.samsung.net/a-mehrotra1/pytorch-asr',
      python_requires='>=3.6.0',
      setup_requires=[
          'git-python'
      ],
      install_requires=[
          'tqdm',
          'numpy',
          'tensorflow',
          'torch==1.7.0',
          'torchaudio==0.7.0',
          'git-python',
          'networkx>=2.5',
          'ctcdecode @ git+https://github.com/parlance/ctcdecode@9a20e00f34d8f605f4a8501cc42b1a53231f1597',
          'torch-edit-distance'
      ],
      dependency_links=[
      ],
      packages=find_packages(where='.', include=[ 'nasbench_asr', 'nasbench_asr.*' ]),
      package_dir={ '': '.' },
      data_files=[],
      cmdclass={
          'build_py': build_maybe_inplace
      }
)
