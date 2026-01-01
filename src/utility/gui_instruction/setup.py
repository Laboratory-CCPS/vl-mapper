from setuptools import find_packages, setup

package_name = 'gui_instruction'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'gui_instruction_node = gui_instruction.gui_instruction_node:main'
        ],
    },
)
