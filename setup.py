from setuptools import find_packages, setup
from typing import List


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

HYPHEN_E_DOT = '-e .'

SRC_REPO = "CategorizeCreditRisk"
__version__ = "0.0.0"
AUTHOR_USER_NAME = "heydido"
AUTHOR_EMAIL = "aashish4.iitd@gmail.com"
REPO_NAME = "CreditRiskSpectrum"


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements.
    """
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A machine learning project to classify credit risk.",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(),
    install_requires=get_requirements(file_path='requirements.txt')
)
