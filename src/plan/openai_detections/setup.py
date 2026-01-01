import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'openai_detections'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'prompts'), glob('prompts/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Josua Lindemann',
    maintainer_email='josua.lindemann@outlook.de',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'openai_detections_node = openai_detections.openai_detections_node:main'
        ],
    },
)
