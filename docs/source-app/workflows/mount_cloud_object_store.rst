:orphan:

######################################################
Mount Data From a Cloud Object Store to the Filesystem
######################################################

**Audience:** Users who want to read files stored in a Cloud Object Bucket in an app.

******************************
Mounting Public AWS S3 Buckets
******************************

=============================
Configuring a Mount in a Work
=============================

To mount data from a cloud bucket to your app compute, initialize a ``Mount`` object with the source path and
the absolute directory path where it should be mounted and pass it to the ``CloudCompute`` it should be mounted on.

.. code:: python
    :emphasize-lines: 9-12

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

=======================
Accessing Mounted Files
=======================

When a mount is configured via ``CloudCompute`` for a ``LightningWork`` running in the cloud, the ``root_dir``
directory path is automatically created and populated with the data before your ``LightningWork`` class even begins
executing. **The files stored in the AWS S3 bucket appear "automagically" as normal files on your local disk**,
allowing you to perform any standard inspection, listing, or reading of them with standard file processing
logic (just like dealing with files on your local machine!)

If we expand on the example above, we can see how you might go about listing and reading a file!

.. code:: python
    :emphasize-lines: 9-14

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

=============================
Mounts & Locally Running Apps
=============================

When running a Lighting App on your local machine, any ``CloudCompute`` configuration (including a ``Mount``)
is ignored at runtime. If you need access to these files on your local disk, you should download a copy of them
to your machine.

===========
Limitations
===========

Currently the following limitations are enforced when using a ``Mount``:

* Mounted files from an S3 bucket are ``read only``. Any modifications, additions, or deletions
  to files in the mounted directory will not be reflected in the cloud object store.
* Mounts can only be configured for a ``LightningWork``. Use in ``LightningFlow`` is not currently supported.
* We enforce a hard limit of ``1,000,000`` objects which we can mount from a particular bucket prefix.
* The maximum size of an object in the bucket can be no more than ``5 GiB``.
* If multiple mounts are configured for a single ``Work``, then they must each specify unique ``root_dir``
  arguments (a unique mount point).

.. note::
   If the bucket prefix contains more than ``1,000,000`` objects or a file greater than ``5 GiB`` in size
   then the ``LightningWork`` will fail to start before it even begins running on the cloud.

----

**********************************************
Mounting Private AWS S3 Buckets - Coming Soon!
**********************************************

We'll Let you know when this feature is ready!

----

************************************************
Mounting Google Cloud GCS Buckets - Coming Soon!
************************************************

We'll Let you know when this feature is ready!
