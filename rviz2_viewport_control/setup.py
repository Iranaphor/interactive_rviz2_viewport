from setuptools import setup
from glob import glob
import os

package_name = 'rviz2_viewport_control'
pkg = package_name

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{pkg}']),
        (f'share/{pkg}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='primordia@live.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            f'boxes.py = {pkg}.boxes:main',
            f'glow_sphere.py = {pkg}.glow_sphere:main',
#            f'transition.py = {pkg}.transition:main',
        ],
    },
)
