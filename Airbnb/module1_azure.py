#def function1(x):
#    return x*x
#function1(2)
#function1([1,2])

#print ("hello")

#cartesian_product = [ [x,y] for x in ['A','B','C'] for y in [1,2,3] ]
#print(cartesian_product)

#LofL = [[.25, .75, .1], [-1, 0], [4, 4, 4, 4]]
#LofL_sum = sum([sum(x) for x in LofL])
#LofL_sum

import urllib.request
import csv
import json 


url = 'https://ussouthcentral.services.azureml.net/workspaces/6336db21218b4e3fbff275874ae2e4ed/services/49f11ae8a1b7445db97f0a1128aa2242/execute?api-version=2.0&details=true'

api_key = 'KiK1vQVniXklw1V25+zzh0pRsId50gu33j6ib0Vi6wg8tAX7oap6uk4AToW7/1ym6IaU/2BvtjpmOae5JgBJkQ=='

headers = {'Content-Type':'application/json', 'Accept':'application/json', 'Authorization':('Bearer '+ api_key)}
countries = ["AU", "CA", "DE", "ES", "FR", "GB", "IT", "NDF", "NL", "other", "PT", "US"]
top_countries = ["NDF", "US", "other", "FR", "IT"]

batch_size = 100
#batch_size = 1

def process_batch(columnnames, listusers, batch_count, batch_number):
	data_old =  {
				"Inputs": 
				{
						"input1":
						{
							"ColumnNames": ["id", "date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser", "country_destination"], 
							"Values": [ user[0:15] + list(" ")]
						}
				},
				"GlobalParameters": 
				{
				}
			}

	data =  {
				"Inputs": 
				{
						"input1":
						{
                            "ColumnNames": columnnames,
							"Values": listusers
						}
				},
				"GlobalParameters": 
				{
				}
			}

	print("**** Batch ****" + str(batch_number) + "/" + str(batch_count))

	body = str.encode(json.dumps(data))
	
	try:
		req = urllib.request.Request(url, body, headers) 
		response = urllib.request.urlopen(req)

		result = response.read()
		#print(result) 
		
		myjson = json.loads(result.decode("utf-8"))
		#print(myjson)

		iUser = 0
		for user_result in myjson['Results']['output1']['value']['Values']:
			user_id = listusers[iUser][0]
			iUser = iUser + 1

			scores = user_result[len(user_result) - 13:len(user_result) - 1]
			#print(scores)

			countries_scores = list(zip(countries, scores))
			countries_scores.sort(key=lambda tup: tup[1], reverse=True)
			countries_scores_top5 = countries_scores[0:5]
			coutries_scores_top5_positive = [x for x in countries_scores_top5 if x[1] != '0']
			countries_printed = []

			for tuple in coutries_scores_top5_positive:
				countries_printed.append(tuple[0])
				#print(user_id + "," + tuple[0])
				f_out.write(user_id + "," + tuple[0] + "\n")
	
			print(countries_printed)
			iTopCountries = 0
			while len(countries_printed) < 5:
				if top_countries[iTopCountries] not in countries_printed:
					countries_printed.append(top_countries[iTopCountries])
					#print(user_id + "," + top_countries[iTopCountries])
					f_out.write(user_id + "," + top_countries[iTopCountries] + "\n")
					
				iTopCountries = iTopCountries + 1

		1+1

	except urllib.HTTPError as error:
		print("The request failed with status code: " + str(error.code))

		# Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
		print(error.info())
		print(json.loads(error.read()))                 

	except:
		print("Unexpected error:", sys.exc_info()[0])
		

with open('C:\\Users\mircean\OneDrive\Airbnb\\results_ml.csv', 'w') as f_out, open('C:\\Users\mircean\OneDrive\Airbnb\\sql_test_users.csv', 'r') as f:
	f_out.write("id,country" + "\n")
	print("id,country")

	data_iter = csv.reader(f, delimiter = ',')
	data = [data for data in data_iter]

	columnnames = data[0][0:15] + ["country_destination"] + data[0][15:]
    
	countusers = 0
	listusers = []

	batch_count = round(len(data)/batch_size)
	batch_number = 0
	for user in data[1:]:
		listusers = listusers + [user[0:15] + [""] + user[15:]]
		if len(listusers) == batch_size:
			process_batch(columnnames, listusers, batch_count, batch_number)
			listusers = []
			batch_number = batch_number + 1

	if len(listusers) > 0:
		process_batch(columnnames, listusers, batch_count, batch_count)

		
		

"""
	data =  {
				"Inputs": 
				{
						"input1":
						{
							"ColumnNames": ["id", "date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser", "country_destination"],
							"Values": [ user[0:15] + list(" ")]
						}
				},
				"GlobalParameters": 
				{
				}
			}
"""

