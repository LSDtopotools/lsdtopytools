"""
Quick scripts to deal with argument parsing for command-line tools.
Why not using argparse or click?
We will have a pretty basic but specific use of argparse and I prefer to debug it internally.
argparse is great, until you have a bug. I had hard time debugging some specific argparse. 
Why am I saying all that again?
Anyway.
B.G.
"""


def get_common_default_param():
	"""
	This function returns a dictionnary of common parameter that all command-line tools will need:
	For example, file name, path , ...
	No arguments.
	Returns:
		The base dictionnary of arguments
	Authors:
		B.G.
	Date:
		01/2019 (last update: 23/02/2019)
	"""
	compam = {}
	compam["file"]="WAWater.bil" # name of our test file.
	compam["path"]="./" # Default is current path
	compam["help"] = False

	return compam

def ingest_param(default_dict, params):
	"""
	Ingest the param and feed the dictionnary. it parses boolean type (just the name of an argument), basic input (e.g.,arg45=621) or comma-separated-values (e.g., cats=henri,garfield,Sigmund)
	Arguments:
		default_dict (python dict): dict containing all the default value for the parameter of that specific analysis
		params (list from sys.argv): the param ran on the command-line tool: 
	Authors:
		B.G.
	Date:
		01/2019 (last update: 23/02/2019)
	"""
	# Let's check all the input params
	for i in range(1,len(params)):
		arg = str(params[i])
		#Checking if this is a bool activator or a argument-bearing param
		if(len(arg.split("=")) == 1):
			# Bool
			default_dict[arg] = True
		elif( "," in arg.split("=")[1]):
			# In that case this argument is a list
			default_dict[arg.split("=")[0]] = arg.split("=")[1].split(",")

		else:
			# ADD HERE ELIF FOR OTHER CASES, for example comma separated lists of basin key
			default_dict[arg.split("=")[0]] = arg.split("=")[1]

	# Done
	return default_dict
