from setuptools import setup

package_name = 'dense_armor_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'dense-armor'],
    zip_safe=True,
    maintainer='Salvatore Pennacchio',
    maintainer_email='tatopenn@gmail.com',
    keywords=['ROS', 'anomaly-detection', 'robotics'],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=(
        'ROS2 node wrapping dense_armor.utility.streaming\'s '
        'MultiChannelStreamingDeviationDetector for real-time, multi-joint '
        'anomaly flagging.'
    ),
    license='BUSL-1.1',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_deviation_node = dense_armor_ros.joint_deviation_node:main',
        ],
    },
)
