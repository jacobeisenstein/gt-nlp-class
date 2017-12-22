# Docker Reference file

This readme is mainly aimed for windows users, but the steps would be similar for linux and mac users as well. Please let me know if you notice any mistakes in the readme. 

A Docker is an open platform that provides the ability to package and run an application in a loosely isolated environment called a container. https://www.docker.com/

* What are docker images?

A Docker image is a read-only template with instructions for creating a Docker container. For example, an image might contain an Ubuntu operating system with Apache web server and your web application installed. You can build or update images from scratch or download and use images created by others. An image may be based on, or may extend, one or more other images. A docker image is described in text file called a Dockerfile, which has a simple, well-defined syntax.

* What is a docker container?

A Docker container is a runnable instance of a Docker image. You can run, start, stop, move, or delete a container using Docker API or CLI commands. When you run a container, you can provide configuration metadata such as networking information or environment variables. Each container is an isolated and secure application platform, but can be given access to resources running in a different host or container, as well as persistent storage or databases. 

You can find more about the docker here: https://docs.docker.com/engine/understanding-docker/. I would suggest you to briefly go through it to get an overview of how things are working..

Pytorch is available for both Linux and Mac users: https://pytorch.org/. So you can go ahead, install them and start working on problem set4. Windows users, however, should follow the steps below to setup an environment in which they could use pytorch. 

* Windows machine: Download and Install the `stable` version of the docker      
*  https://docs.docker.com/docker-for-windows/install/  
*  All the steps for installation are clearly mentioned in the above link clearly.

### PART-I
Once, you have downloaded and installed the docker, follow the steps below to get the docker up and running for your problem set 4. Please follow the steps carefully. Images have also been attached below for better guidance.
- Start the docker app.
- ![](images/1.png?raw=true "")
- Then run the following command:
  - ```docker pull bmurali1994/nlp:pset4``` 
  - ![](images/2.png?raw=true "")
  - The command above basically pulls the docker image from the docker hub 
     (https://hub.docker.com/r/bmurali1994/nlp/)
  - The new image: bmurali1994/nlp:pset4 would be downloaded and extracted. 
  - ![](images/3.png?raw=true "")
  - ![](images/4.png?raw=true "")
  - run the following command to make sure the image is downloaded properly:
    - ```docker images``` 
    - You should find the image ```bmurali1994/nlp``` listed there. 
    - ![](images/18.png?raw=true "")
- Now, navigate to the directory where you want to work on problem set4. Create a new directory to store all the files.
  - ```mkdir pset4```
  - ![](images/5.png?raw=true "")
- Then, run the following command:
  - ```docker run -d -p 8888:8888 --name nlp -w /opt/pset4 -v `pwd`/pset4:/opt/pset4 bmurali1994/nlp:pset4 sh -c "jupyter notebook --no-browser --ip=0.0.0.0 --port=8888"```
  - ![](images/6.png?raw=true "")
  - Sometimes there might be issues with parsing your username of the computer if it has spaces. Then replace ``` `pwd` ``` with the full directory address. For example in my case, the username is ```Murali Raghu Babu.B```. So, I replace the ``` `pwd` ``` with ```/c/Users/Murali\ Raghu\ Babu.B```. The command would then be the following:
  - ```docker run -d -p 8888:8888 --name nlp -w /opt/pset4 -v /c/Users/Murali\ Raghu\ Babu.B/pset4:/opt/pset4 bmurali1994/nlp:pset4 sh -c "jupyter notebook --no-browser --ip=0.0.0.0 --port=8888"```
  - ![](images/7.png?raw=true "")
  - This command basically does the following:
    - runs the image in a container named ```nlp``` in daemon mode
    - maps the 8888 port of your machine to 8888 of the docker
    - sets your working directory within docker as ```/opt/pset4```
    - maps your newly created ```pset4``` folder to the ```/opt/pset4``` folder of the docker and 
    - starts the jupyter notebook within the docker with proper network settings.
- Then run the command:
  - ```docker ps -a```
  - You should see the container named ```nlp``` running.
  - ![](images/8.png?raw=true "")
- Then run the following command:
  - ```docker logs nlp```
  - ![](images/9.png?raw=true "")
  - This command will now print the url along with the token at which the jupyter notebook is running. 
  - Copy paste the url into your browser and you should see the jupyter notebook running. 
  - Your url along with the token looks something like this.
     (```http://0.0.0.0:8888/?token=add497844b4943e1951b89dc6b307451f29c808de14a9```) 
  - Be careful when copy pasting your url, it should not have any spaces in between.
- The url on copy pasting into a browser will not work for a windows machine. So additionally windows users, run this command:
  - ```docker-machine ip default```
  - The above command gets the ip address of the host machine.
  - ![](images/10.png?raw=true "")
  - Now replace the 0.0.0.0 with the ip address obtained above and paste the entire url into your browser.
  - (```http://192.168.99.100:8888/?token=add497844b4943e1951b89dc6b307451f29c808de14a9```)
  - ![](images/11.png?raw=true "")
- Now, you're all set to go!!
  - Copy paste the entire problem set folder/files into the newly created folder ```pset4```. 
  - You should be able to access these files from the browser and run them.
  - ![](images/12.png?raw=true "")
  - ![](images/13.png?raw=true "")

### PART-II
- For running the unit test files, we have the entire environment running inside the docker. So, we need to execute the ```nosetests``` command inside the docker. To do this, follow the steps below:
  - Initially run the command:
  - ```docker exec -it nlp bash```
  - ![](images/14.png?raw=true "")
  - This command will start a bash session in the container. 
  - Your working directory will then be ```/opt/pset4``` as we have set in the commands before.
  - Navigate to the problem set files inside and run the following command:
    - ![](images/15.png?raw=true "")
    - ```nosetests -v tests/test_parser.py```
    - The tests would run and you will be able to see your result on the standard output.
    - ![](images/16.png?raw=true "")
    - ![](images/17.png?raw=true "")
    - Run the command ```exit``` to exit the bash session.
- You can keep making changes in the python files on your browser and run them there to check the output. 
- To test them on the unit tests, you need to start a bash session in the container and run the command shown above.

Please feel free to get back to me if you have further questions or have some other issues. 

Additional links that might help:-
- https://www.docker.com/what-docker
- https://docs.docker.com/docker-for-windows/
- https://docs.docker.com/engine/reference/run/
- https://github.com/prakhar1989/docker-curriculum/issues/27
- https://docs.docker.com/engine/reference/commandline/exec/
