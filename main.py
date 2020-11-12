
from business import BusinessManager
import model


def main():
	# 取数据
	mgr = BusinessManager('data/')

	# 生成训练模型需要的train_data, test_data
	train_data, test_data = mgr.generate_model_data()
	#
	feature_builder = model.FeatureBuilder('tfidf')
	X_train, y_train, X_test, y_test = feature_builder.get_feature(train_data,\
																   test_data)

	lrmodel = model.LinearModel()
	lrmodel.train(X_train, y_train)
	lrmodel.save(model_path)

	mgr.set_sentiment_model(lrmodel)

	business_ids = mgr.get_business_ids() # return list
	for bid in business_ids:
		summary = mgr.aspect_based_summary(bid)
		print(summary)


if __name__ == "__main__":
	main()


