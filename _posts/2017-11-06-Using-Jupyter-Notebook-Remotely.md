---
layout:     post
title:      "Using Jupyter Notebook on Server"
subtitle:   ""
date:       2017-11-06 12:00:00
author:     "Siqi"
header-img: "img/default.jpg"
mathjax: true
tags:
    - jupyter
---

When you are working on large data set or need extra computing power, you may need to do the work remotely on a server. For students, since you have access to AWS educate, you should definitely take advantage of it.

## Steps

### On the Server

First, you may want to set up a password for your notebook. For detailed instruction, you can look up [here](http://jupyter-notebook.readthedocs.io/en/stable/public_server.html#notebook-server-security).  

If you are looking for a quick set up, you can follow these steps:  
```bash
jupyter notebook --generate-config
```
Then you can create your password by
```bash
jupyter notebook password
```

After setting up password, start a jupyter notebook in no browser mode, and specify a port you like. For example,   
```bash
jupyter notebook --no-browser --port=8889
```
Of course, you can replace 8889 with any port you like, just make sure there is no clash.  
If you plan to run the notebook for long period of time, you can run your notebook without hangup  
```
nohup jupyter notebook --no-browser --port=8889
```
In this way, your notebook will keep running even if you log out of the server.  


### On Local Machine
Now you need to connect to your notebook from your local machine.  
You need to create a ssh tunnel to your server and bind your notebook port to a specified local port. Continuing from the previous example, on your local machine, 
```bash
ssh [USER]@[SERVER] -L 8889:localhost:8888 -N
```
where [USER] is your user name, and [SERVER] is your server's address.  
Now you can launch jupyter notebook as usual, by typing 'localhost:8888' in your favorite browser.  

### To Kill the Jupyter Notebok Process
Since we launched jupyter notebook with ```nohup```, you can do the following to kill the jupyter notebook process. 
```bash
netstat -tulpn
```
Then ```kill``` the pid with python.  