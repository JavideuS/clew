from setuptools import find_packages, setup

package_name = "fleet_coordinator"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/coordinator.launch.py"]),
        (
            "share/" + package_name + "/config",
            ["config/mission.example.yaml", "config/params.example.yaml"],
        ),
    ],
    install_requires=["setuptools", "requests", "pyyaml"],
    zip_safe=True,
    maintainer="Javier Gonzalez Villasmil",
    maintainer_email="javi.rm2005@gmail.com",
    description=(
        "Decoupled-mode multi-robot fleet coordinator: Spooky global "
        "planning + release-gated per-robot dispatch."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "coordinator_node = fleet_coordinator.coordinator_node:main",
        ],
    },
)
