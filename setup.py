from setuptools import find_packages, setup
from typing import List
import pandas as pd

minus_e_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirments]

        if minus_e_dot in requirments:
            requirments.remove(minus_e_dot)

    return requirments


setup(
    name='mlproject',
    version='0.0.1',
    author='Somak Banerjee',
    author_email='somak.banerjee.work@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)