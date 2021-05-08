from setuptools import setup
import os
dirpath = os.path.dirname(__file__)
command='cd {}/{}; ./install.sh'.format(dirpath,installData)
try:
    os.system(command)
except Exception as e:
    print("failed to install pre-req for tensorrt",str(e))
    
setup( 
    name='objectDetectionSSD',
    
    version='0.1',
    description='Mobilenet ssd object detection',
    url='http://demo.vedalabs.in/',

    # Author details    
    author='Kumar',
    author_email='tech@gmail.com',

    packages=['objectDetectionSSD'],
    package_data={'objectDetectionSSD': ['labels.json','installData']},
    install_requires=['tensorflow'] ,

    zip_safe=False
    )