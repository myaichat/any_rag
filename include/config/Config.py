import os, sys, json, codecs
import re
from datetime import datetime
from os.path import join


from os.path import isfile


e=sys.exit

class MutableAttribute:
	def __init__(self):
		self.parent = None
		self.name = None
		self.real_name = None

	def __set_name__(self, owner, name):
		self.name = f"_{name}"
		self.real_name = name

	def __get__(self, obj, objtype=None):
		if self.parent is None:
			self.parent = obj
		return getattr(obj, self.name, None)

	def __set__(self, obj, value):
		if self.parent is None:
			self.parent = obj
		processed_value = self.process(value)
		setattr(obj, self.name, processed_value)
		self.notify_change(processed_value)

	def process(self, value):
		print('777 Processing:', self.real_name, value)
		if hasattr(self.parent, 'process'):
			return self.parent.process(self.real_name, value)
		return value

	def notify_change(self, value):
		pub.sendMessage(f'{self.real_name}_changed', value=value)
		print('888 Notifying:', self.real_name, value)
		#pub.sendMessage('{self.real_name}_changed', name=self.real_name, value=value)





class NotifyingDict(dict):
	def __init__(self, *args, parent=None, key=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.parent = parent
		self.key = key
		self._processing = False
		for k, v in self.items():
			if isinstance(v, dict):
				self[k] = NotifyingDict(v, parent=self, key=k)

	def __setitem__(self, key, value):
		if isinstance(value, dict) and not isinstance(value, NotifyingDict):
			value = NotifyingDict(value, parent=self, key=key)
		super().__setitem__(key, value)
		self.propagate_change()
		

	def __getattr__(self, name):
		try:
			
			return self[name]
		except KeyError:
			raise AttributeError(f"'NotifyingDict' object has no attribute '{name}'")

	def __setattr__(self, name, value):
		if name in ['parent', 'key', '_processing']:
			super().__setattr__(name, value)
		else:
			self[name] = value
		

	def propagate_change(self):
		if self.parent and not self._processing:
			if isinstance(self.parent, NotifyingDict):
				self.parent.propagate_change()
			elif isinstance(self.parent, MutableDictAttribute):
				self.parent.child_changed()

class MutableDictAttribute:
	def __init__(self):
		self.parent = None
		self.name = None
		self.real_name = None


	def __set_name__(self, owner, name):
		self.name = f"_{name}"
		self.real_name = name

	def __get__(self, obj, objtype=None):
		if self.parent is None:
			self.parent = obj
		
		return getattr(obj, self.name, None)

	def __set__(self, obj, value):
		if self.parent is None:
			self.parent = obj
		processed_value = self.process(value)
		if isinstance(processed_value, dict):
			processed_value = NotifyingDict(processed_value, parent=self, key=self.real_name)
		setattr(obj, self.name, processed_value)
		

	def process(self, value):
		
		if hasattr(self.parent, 'process'):
			return self.parent.process(self.real_name, value)
		
		return value

	def child_changed(self):
		if hasattr(self.parent, 'process'):
			current_value = getattr(self.parent, self.name, None)
			if current_value is not None:
				current_value._processing = True
				processed = self.parent.process(self.real_name, current_value)
				current_value._processing = False
				setattr(self.parent, self.name, processed)

class Config(object): 
	prompt_log = MutableDictAttribute()
	def __init__(self, **kwargs):
		self.home=None
		self.dump_file={}
		self.cfg={}
		self.mta=set()
		self.clients={}
		self.apis={}
		# Get the current timestamp
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.prompt_log = self.get_attr('prompt_log',{},join('log',f'.prompt_log_{timestamp}.json'))

	def get_client (self,api):
		clients=self.clients
		if api not in clients:
			client_api = self.apis[api].AsyncClient
			api_key = os.getenv(f"{api.upper()}_API_KEY")
			assert api_key, f"API key for '{api.upper()}_API_KEY' not found"
			clients[api] =  client_api(api_key)

		return clients[api]

	def process(self, attr_name, value):
			#print   ('-----Processing:', attr_name, value)
			if attr_name in self.mta: # ['page_id', 'reel_id', 'user_token','followers_count','uploaded_cnt']:
				#print(f"Parent processing: {attr_name} = {value}")
				if value:
					self.set_attr(attr_name, value)
				return value
			
			return value 
						


	def get_attr(self, attr, default=None, dump_file='.config.json'): 
		if attr not in self.dump_file:
			self.dump_file[attr]=dump_file
		config_fn=self.dump_file[attr]
		self.mta.add(attr)
		print('-------------------config_fn: ' , attr, config_fn)
		if config_fn not in self.cfg:
			self.cfg[config_fn]={}
		cfg=self.cfg[config_fn]

		if not cfg:
			if isfile(config_fn):
				try:
					print(f"Reading config file {config_fn}")
					with open(config_fn, 'r') as f:
						content = f.read().strip()
						#pp(content)
						if content:
							cfg_dump = json.loads(content)
							#pp(cfg_dump)
							self.cfg[config_fn]=cfg=cfg_dump
						else:
							print(f"Warning: {config_fn} is empty.")
				except json.JSONDecodeError as e:
					print(f"Error reading config file {config_fn}: {e}")
					#print("Initializing with an empty PropertyDefaultDict.")
				except Exception as e:
					print(f"Unexpected error reading config file {config_fn}: {e}")
					#print("Initializing with an empty PropertyDefaultDict.")
			else:
				print(f"Warning: connfig file {config_fn} does not exist.")
			
				
		if cfg:
			print(8888, cfg)
			#print (attr.name)
			value=cfg.get(attr, default)
			print('Getting:', attr, type(value))   
		
			
			return value
		self.cfg[config_fn]=cfg
		return default
	def set_attr(self, attr, value):
		#print('Setting:', attr, value, type(value))
		assert attr in self.dump_file, f'set_attr: No dump file specified for attr "{attr}"'
		dump_file = self.dump_file[attr]   
		assert dump_file, f'set_attr: dump_file is not set  for attr "{attr}"'     
		cfg=self.cfg[dump_file]
		#pp(self.cfg)
		assert cfg is not None, dump_file
		cfg[attr]=value

		assert dump_file, 'set_attr: No dump file specified'
		print('Dumping ******************************:', attr, dump_file)    
		with open(dump_file, 'w') as f:
			json.dump(cfg, f, indent=2)
		

		