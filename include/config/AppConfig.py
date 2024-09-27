
from include.config.Config import Config

import logging

log=logging.getLogger()



class AppConfig(Config): 
	def __init__(self):
		
		self.gid=0
		
		self.all={}
	def init(self, **kwargs):
		Config.__init__(self,**kwargs)




		
		
			