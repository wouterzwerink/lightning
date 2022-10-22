:orphan:

##############################
Mount Data From a Cloud Bucket
##############################

**Audience:** Users who want to read files stored in a Cloud Object Bucket in an app.

******************************
Mounting Public AWS S3 Buckets
******************************

=============================
Configuring a Mount in a Work
=============================

To mount data from a cloud bucket to your app compute, initialize a ``Mount`` object with the source path
of the s3 bucket and the absolute directory path where it should be mounted and pass the ``Mount`` to the
``CloudCompute`` of the ``LightningWork`` it should be mounted on.

In this example, we will mount an S3 bucket: ``s3://ryft-public-sample-data/esRedditJson/`` to ``/content/esRedditJson/``.

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
                       mount_path="/content/esRedditJson/",
                   ),
               )
           )

       def run(self):
           self.my_work.run()

You can also pass multiple mounts to a single work by passing a ``List[Mount(...), ...]`` to the
``CloudCompute(mounts=...)`` argument.

.. note::

    * We enforce a hard limit of ``1,000,000`` objects which we can mount from a particular bucket prefix.
    * The maximum size of an object in the bucket can be no more than ``5 GiB``.
    * If multiple mounts are configured for a single ``LightningWork``, then they must each specify unique
      ``mount_path`` arguments (a unique mount point).

==============================
Accessing Files From The Mount
==============================

Once a ``Mount`` object is passed to ``CloudCompute``, you can access, list, or read any file from the mount
under the specified ``mount_path``, just like you would if it was on your local machine.

Assuming you mount_path is ``"/content/esRedditJson/"`` you can do the following:

-----------
List Files:
-----------

.. code:: python

    files = os.listdir("/content/esRedditJson/")

-----------
Read Files:
-----------

.. code:: python

    with open("/content/esRedditJson/esRedditJson1", "r") as f:
        some_data = f.read()

    # do something with "some_data"...

---------------------
See the Full Example:
---------------------

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
                       mount_path="/content/esRedditJson/",
                   ),
               )
           )

       def run(self):
           self.my_work.run()

The ``LightningWork`` component in the code above (``MyWorkClass``) would print out a list of files stored
in the mounted s3 bucket & then read the contents of a file ``"esRedditJson1"``.

.. note::

    When running a Lighting App on your local machine, any ``CloudCompute`` configuration (including a ``Mount``)
    is ignored at runtime. If you need access to these files on your local disk, you should download a copy of them
    to your machine.

.. note::

    Mounted files from an S3 bucket are ``read-only``. Any modifications, additions, or deletions
    to files in the mounted directory will not be reflected in the cloud object store.

===========
Limitations
===========

Currently the following limitations are enforced when using a ``Mount``:

* Mounted files from an S3 bucket are ``read-only``. Any modifications, additions, or deletions
  to files in the mounted directory will not be reflected in the cloud object store.
* Mounts can only be configured for a ``LightningWork``. Use in ``LightningFlow`` is not currently supported.
* We enforce a hard limit of ``1,000,000`` objects which we can mount from a particular bucket prefix.
* The maximum size of an object in the bucket can be no more than ``5 GiB``.
* If multiple mounts are configured for a single ``Work``, then they must each specify unique ``mount_path``
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
