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
The Amazon Web Service (AWS) Simple Storage Solution (S3) is a fundamental piece of cloud infrastructure
which acts as an infinitely scalable object store. It is commonly used by organization who wish to
keep vast amounts of data files available for use by their applications which run on the cloud.

Utilizing AWS S3 within components of a Lighting App allows the application to natively load
arbitrary files from S3 into the application; this might be training data, model checkpoints,
or configuration files (among many other use cases).

----

***********
Limitations
***********

Currently the following limitations are enforced when using a ``Mount``:

* Mounted files from an S3 bucket are ``read only``. Any modifications, additions, or deletions
  to files in the mounted directory will not be reflected in the cloud object store.
* Mounts can only be configured for a ``LightningWork``. Use in ``LightningFlow`` is not currently supported.
* Only public S3 buckets can be mounted.
* We enforce a hard limit of ``1,000,000`` objects which we can mount from a particular bucket prefix.
* The maximum size of an object in the bucket can be no more than ``5 GiB``.
* If multiple mounts are configured for a single ``Work``, then they must each specify unique ``root_dir``
  arguments (a unique mount point).

.. warning::
   If the bucket prefix contains more than ``1,000,000`` objects or a file greater than ``5 GiB`` in size
   then the ``LightningWork`` with the ``Mount`` configured on it will fail before it begins running on the cloud.

----

*****************************
Configuring a Mount in a Work
*****************************

In order to configure a mounted s3 bucket in a work, you simply set the ``mounts`` field of a ``CloudCompute``
object which is passed to any ``LightingWork`` class. The ``source`` field indicates the bucket prefix (i.e. folder)
to use as the data source, and the ``root_dir`` argument is used to specify where the data should appear on the
filesystem disk.

.. code-block::python

    import lightning as L
    from lightning_app import CloudCompute
    from lightning_app.storage import Mount

    class Flow(L.LightningFlow):
       def __init__(self):
           super().__init__()
           self.my_work = MyWorkClass(
               cloud_compute=CloudCompute(
                   mounts=Mount(
                       source="s3://ryft-public-sample-data/esRedditJson/",
                       root_dir="/content/esRedditJson/",
                   ),
               )
           )

       def run(self):
           self.my_work.run()

You can also pass multiple mounts to a single work by passing a ``List[Mount(...), ...]`` to the
``CloudCompute(mounts=...)`` argument.

----

***********************
Accessing Files on Disk
***********************

When a mount is configured in for a ``LightningWork`` running in the cloud, the ``root_dir`` directory
path is automatically created and populated with the data before your ``LightningWork`` class even begins
executing. **The files stored in the AWS S3 bucket appear "automagically" as normal files on your local disk**,
allowing you to perform any standard inspection, listing, or reading of them with standard file processing
logic (just like dealing with files on your local machine!)

If we expand on the example above, we can see how you might go about listing and reading a file!

.. code-block::python

    import os

    import lightning as L
    from lightning_app import CloudCompute
    from lightning_app.storage import Mount

    class MyWorkClass(L.LightningWork):
       def run(self):
           files = os.listdir("/content/esRedditJson/")
           for file in files:
               print(file)

           with open("/content/esRedditJson/esRedditJson1", "r") as f:
               some_data = f.read()
               # do something with "some_data"...

    class Flow(L.LightningFlow):
       def __init__(self):
           super().__init__()
           self.my_work = MyWorkClass(
               cloud_compute=CloudCompute(
                   mounts=Mount(
                       source="s3://ryft-public-sample-data/esRedditJson/",
                       root_dir="/content/esRedditJson/",
                   ),
               )
           )

       def run(self):
           self.my_work.run()

The ``LightningWork`` component in the code above (``MyWorkClass``) would print out a list of files stored
in the mounted s3 bucket & then read the contents of a file ``"esRedditJson1"``.

____

*****************************
Mounts & Locally Running Apps
*****************************

When running a Lighting App on your local machine, any ``CloudCompute`` configuration (including a ``Mount``)
is ignored at runtime. If you need access to these files on your local disk, you should download a copy of them
to your machine.
