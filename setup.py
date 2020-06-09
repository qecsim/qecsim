from setuptools import setup, find_packages
import qecsim

setup(
    name='qecsim',
    version=qecsim.__version__,
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3',
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'qecsim = qecsim.cli:cli',
        ],
    }
)
