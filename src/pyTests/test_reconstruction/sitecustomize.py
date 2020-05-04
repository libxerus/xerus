# If this files directory is on sys.path, this file will be loaded when Python starts up.
import os, site
site.addsitedir(os.path.dirname(__file__))
