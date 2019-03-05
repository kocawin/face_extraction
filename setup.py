from setuptools import setup, find_packages

setup(
    name='face-extraction',
    version='.1',
    packages=find_packages(exclude=['tests*', 'build', 'dist']),
    license='MIT',
    description='Face extractor from videos',
    long_description=open('README.md').read(),
    install_requires=[
        'face_recognition',
        'opencv-python',
        'numpy',
        'dlib',
        'click'
    ],
    url='https://github.com/rockkoca/face-extraction',
    author='KocA',
    author_email='',
    python_requires='>=3.6',
)
