from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sentinel_control'

setup(
    name=package_name,              # NOT 'sentinel-control'
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maciej',
    maintainer_email='maciekmularczyk@op.pl',
    description='Sentinel control package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_obstacles = sentinel_control.map_obstacles:main',
            'navigate_bt   = sentinel_control.navigate_bt:main',
            'path_planner  = sentinel_control.path_planner:main',
        ],
    },
)

