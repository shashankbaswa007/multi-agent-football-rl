"""
Setup script for Multi-Agent Football RL package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='marl_football',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Multi-Agent Reinforcement Learning for Football/Soccer',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/multi_agent_football',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
        ],
        'grf': [
            'gfootball>=2.10',
        ],
        'distributed': [
            'ray[rllib]>=2.6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'marl-train=training.train_ppo:main',
            'marl-replay=visualization.replay_viewer:main_replay',
            'marl-heatmap=visualization.heatmap:main_heatmap',
            'marl-passnet=visualization.pass_network:main_pass_network',
            'marl-plots=visualization.training_plots:main_plots',
        ],
    },
    include_package_data=True,
    package_data={
        'marl_football': ['configs/*.yaml'],
    },
    zip_safe=False,
)