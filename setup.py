from setuptools import setup, find_packages

setup(
    name='luna_project',
    version='0.2.0',
    author='Syra Team',
    author_email='your.email@example.com',
    description='Sistema de diálogo adaptativo e dinâmico LunaGPT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0,<5.0.0',  # Cap version to avoid breaking changes
        'tokenizers>=0.13.0',
        'tqdm>=4.65.0',
        'regex>=2023.6.3',
        'numpy>=1.24.0',
        'sentencepiece>=0.1.99',
        'peft>=0.4.0',
        'evaluate>=0.4.0',
        'datasets>=2.12.0',
        'accelerate>=0.20.0',
        'pdfminer.six>=20221105',
        'scikit-learn>=1.2.2',
        'sentence-transformers>=2.2.2',
        'PyPDF2>=3.0.0',
        'python-docx>=0.8.11',
        'beautifulsoup4>=4.11.1',
        'requests>=2.28.1',
        'pandas>=1.5.0',
        'epub>=0.5.2',  # Opcional para processar ebooks
        'psutil>=5.9.0',  # For hardware monitoring
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'lunagpt=main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)