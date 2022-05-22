from setuptools import setup, find_packages

setup(
    name='CRP_backend',
    version='0.1.0',    
    description='Backend for Concept Relevance Propagation',
    author='Reduan Achtibat',
    license='GNU GPLv3',
    packages=find_packages(),
    install_requires=[
        'zennit',
        'zennit-crp',
        'sklearn', #?
        'scikit-learn', #?
        'scipy',
        'opencv-python',
        'flask',
        'flask-socketio',
        'eventlet', # deployment server for sockets
        'celery'
    ],
   # extras_require=[
    #   'Pillow-SIMD' 
    #],
    python_requires='>=3.8',

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
    ],
)