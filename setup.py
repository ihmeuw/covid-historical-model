import os

from setuptools import setup, find_packages


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "covid_historical_model", "__about__.py")) as f:
        exec(f.read(), about)

    # with open(os.path.join(base_dir, "README.rst")) as f:
    #     long_description = f.read()

    install_requirements = [
        'click',
        'covid_shared>=1.0.54',
        'drmaa',
        'dill',
        'loguru',
        'numpy',
        'scipy',
        'pandas',
        'xspline',
        'tables',
        'xlrd',
        'openpyxl',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'tqdm',
        'pypdf2',
        'pyyaml',
        'ipython',
        'jupyter',
        'jupyterlab'
    ]

    test_requirements = [
        'pytest',
    ]

    doc_requirements = []

    internal_requirements = [
        'db_queries',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        # long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            'docs': doc_requirements,
            'test': test_requirements,
            'internal': internal_requirements,
            'dev': [doc_requirements, test_requirements, internal_requirements]
        },

        # entry_points={'console_scripts': [
        #     'XXX=covid_historical_model.cli:XXX'
        # ]},
        zip_safe=False,
    )
