:orphan:

########################################
Mount an AWS S3 Bucket to the Filesystem
########################################

.. note:: The contents of this page is still in progress!

**Audience:** Users who want to read files stored on an AWS S3 Bucket in an app.

----

****************************
Why Do I Need To Use AWS S3?
****************************
In a Lightning App some components can be executed on their own independent pieces of hardware.
The Amazon Web Service (AWS) Simple Storage Solution (S3) is a fundumental piece of cloud infrastructure
which acts as an infinitely scalable object store. It is commonly used by organization who wish to
keep vast amounts of data files available for use by their applications which run on the cloud.

Utilizing AWS S3 within components of a Lighting App allows the application to natively load
arbitrary files from S3 into the application; this might be training data, model checkpoints,
or configuration files (among many other use cases).

----

***********
Limitations
***********


----

*************************
Passing a Mount to a Work
*************************

----

***********************
Accessing Files on Disk
***********************

----

*********************
Using Multiple Mounts
*********************
