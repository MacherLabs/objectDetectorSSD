from setuptools import setup

setup( 
    name='objectDetectionSSD',
    
    version='0.1',
    description='Mobilenet ssd object detection',
    url='http://demo.vedalabs.in/',

    # Author details    
    author='Kumar',
    author_email='tech@gmail.com',

    packages=['objectDetectionSSD'],
    package_data={'objectDetectionSSD': ['labels.json']},
    install_requires=['tensorflow==1.15.2'] ,

    zip_safe=False
    )
