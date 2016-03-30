from django.shortcuts import render
from CoreFunctions import StockMarketPrediction
import os
# Create your views here.

def home(request):
	context = {}
	return render(request,"index.html",context)

def PredictionResults(request):

	if request.method == "POST":

		stock_name = request.POST.get("stock_symbol")
		trading_day = request.POST.get("trading_day")

		request.session["stock"] = stock_name
		request.session["trading_day"] = trading_day

		trading_day = int(trading_day)		
		accuracy, output, current_data,precision,recall, specificity= StockMarketPrediction.main(stock_name,trading_day)

	context = {
		"accuracy" : accuracy,
		"output" : output,
		"current_data": current_data,
		"trading_day": trading_day,
		"stock_name": stock_name,
		"precision": precision,
		"recall": recall,
		"specificity": specificity
	}

	return render(request, "result.html",context)
	

