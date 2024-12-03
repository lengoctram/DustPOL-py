import os
from setuptools import setup, find_packages

setup(
    name="DustPOL_py",  # The name of your package
    version="0.1.6",  # Initial version of the package
    author="Le N. Tram",
    author_email="lengoctramlyk31@gmail.com",
    description="modeling dust polarization",
    long_description=os.path.realpath('__file__'),#open("README.md").read(),  # Use README.md as long description
    long_description_content_type="text/markdown",
    url="https://github.com/lengoctram/DustPOL-py",  # Replace with your project's URL
    packages=find_packages(),  # Automatically find all packages and sub-packages
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[  # List your project's dependencies here
        "numpy==1.20.3",
        "matplotlib",
        "astropy",
        "scipy",
        "joblib",
        "pwlf"
        # Add other dependencies as needed
    ],
    include_package_data=True,  # To include non-code files specified in MANIFEST.in
    # entry_points={  # Optional: specify console scripts if your package has CLI commands
    #     "console_scripts": [
    #         "my_project=my_project.main:main_function",  # command=package.module:function
    #     ],
    # },
)
