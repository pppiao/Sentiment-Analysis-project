# import nltk
import os
from collections import defaultdict
import yaml
import json
from collections import Counter
import nltk
import random


# todo: 提取一个句子的aspects集合
def extract_aspects(sent):
	"""
	从一个句子的review中抽取aspects
	:param
	:param sent:传入的句子
	:return:
	"""
	aspects = {}
	for word, tag in nltk.pos_tag(nltk.word_tokenize(sent)):
		if tag == "NN":
			aspects.add(word)
	return aspects


# TODO 构建Review类
# 固有属性：文本，评分，用户id, 商品id
class Review:
	def __init__(self, data):
		self.user_id = data["user_id"]
		self.business_id = data["business_id"]
		self.stars = data["stars"]
		self.text = data["text"]

	def compute_doc_aspects(self, doc, topk=5):
		"""
		针对一个business，提取其topk个aspects
		:param self:
		:param doc: business对应的review数据
		:param topk: aspects的数目
		:return: topk个aspects
		"""
		sents = []
		for line in doc:
			# doc是一个段落的集合？？，需要对其分句，- 所有的reviews集合组成的一个段落为doc
			# 思路，doc -->分句，
			# nltk.sent_tokenize的分句依据？？智能分割句子，逗号、分号不是句子
			sents_ = nltk.sent_tokenize(line)
			sents.extend(sents_)

		topic = Counter()
		# Counter（计数器）是对字典的补充，用于追踪值的出现次数。
		for sent in sents:
			aspects_= extract_aspects(self, sent)
			topic.update(aspects_)
		aspects, freq = zip(*topic.most_common(topk))
		# zip -返回一个对象
		"""
			--- 找出一个序列中出现次数最多的元素 ----
			collections.Counter 类
			most_common() 方法
		"""
		return aspects


# TODO 构建BusinessManager类，
	#  明确其属性data, aspects, user_stars, sentiment_model
	# 	方法：获取business_id, 获取针对该business_id对应的reviews，
	# 	构建该business_id对应的business的aspects，
	# 获取business的score
class BusinessManager(object):

	def __init__(self, init_dir=None):
		"""
		类的构造函数或初始化方法
		:param init_dir: business.json的地址
		第一种方法__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，
		当创建了这个类的实例时就会调用该方法
		self 代表类的实例，self 在定义类的方法时是必须有的，
		虽然在调用时不必传入相应的参数。
		TP--
			这BusinessManager类的固有属性有：data, aspects, user_stars, sentiment_model
		"""

		self.data = defaultdict(list) # 构建一个默认value为list的字典
		self.aspects = defaultdict(list)
		self.user_stars = {}
		self.sentiment_model = None
		if init_dir:
			self.load_data(init_dir)

	def load_data(self, review_dir):
		"""
		获取所有business下的review文本数据
		:param review_dir: 评论文件的地址
		:return:
		"""
		# business data
		# review data
		user_stars = defaultdict(float)
		# 列出该路径下存在的文件、文件夹, review_dir下包含了business
		# 的review文件
		for review_file in os.listdir(review_dir):
			review_path = os.path.join(review_dir, review_file)
			review_data = json.load(open(review_path, "r", encoding="utf-8"))

			# todo：**用法不明白**--- 类之间的交互
			review = Review(review_data)

			business_id = review.business_id # id
			business = self.data.get(business_id) # data是一个default(list)获取business的名称，使用business_id作为名称，因为business_id是唯一的
			business.append(review) #???
			# business是一个列表 [business_id值， review1, review2，...]

			user_stars[review.user_id] += review.stars  #????
			# 应该是求针对该business,所有user 对该商品的stars之和
		self.user_stars = {user_id: stars/len(user_stars) \
						   for user_id, stars in user_stars.items()}

	def get_business_ids(self):
		return list(self.data.keys())

	def get_business_reviews(self, business_id):
		return self.data.get(business_id, [])
	"""
	# 字典的get()用法,get(key, default)
    key -- 字典中要查找的键。
    default -- 如果指定键的值不存在时，返回该默认值。
	"""

	def load_aspects(self, aspect_config):
		assert os.path.exists(aspect_config)
		self.aspects = yaml.safe_load(aspect_config)
		# yaml.safe_load用于读取.yaml文件中的数据

	def build_aspects(self):
		for business_id, reviews in self.data.items():
			doc = [review.text for review in reviews]
			# review.text????
			self.aspects[business_id] = compute_doc_aspects(doc, topk=5)

	def get_business_aspects(self, business_id):
		if business_id not in self.aspects:
			print("not find business_id")
			return []
		return self.aspects[business_id]

	def get_all_reviews(self):
		return [review for review in reviews for reviews in list(self.data.values())]

	def get_business_score(self, business_id):
		# Compute the score for each business
		reviews = self.data[business_id]
		print("the type of review is :", type(reviews[0]))
		print(" the reviews is a list, for example, reviews[0] and reviews[1] are: \n")
		print(reviews[0], '\n', reviews[1])

		scores = [review.stars for review in reviews]
		ave_score = sum(scores) / len(scores)
		return ave_score

	def get_user_score(self, user_id):
		# Get the specific user's score for all businesses
		# 获取每个用户的打分均值，用于作为其评分分类的标准
		reviews = self.get_all_reviews()
		scores = [review.stars for review in reviews if review.user_id==user_id]
		ave_score = sum(scores) / len(scores)
		return ave_score

	def get_aspect_summary(self, business_id, aspect):
		"""
		根据传入的参数business_id ，找出对应的reviews；
		根据传入的参数aspect 和 reviews, 找出对应的review
		:param business_id:
		:param aspect:
		:return:
		"""
		# Get the summary review of a business
		stars = 0.0
		pos_sents, neg_sents = [], []

		reviews = self.data[business_id]
		for review in reviews:
			stars += review.stars
			if not review.text.contains(aspect):
				continue

			review_segment = get_segment(review, aspect)
			# get_segment：一个review可能会包含对多个aspect的评论，
			# 是针对当前aspect,提取评论内容
			# 进而，获取其情感分数，判定评论的属性neg/POS
			score = self.sentiment_model.predict(review_segment)

			if score > threshold:
				pos_sents.append(review_segment)
			else:
				neg_sents.append(review_segment)

		stars = stars / (len(pos_sents) + len(neg_sents))

		return dict(rating=stars, pos=pos_sents, neg=neg_sents)

	def aspect_based_summary(self, business_id):
		"""
		返回一个business的summary
		:param business_id: 指定的business
		:return:
		"""
		aspect_summary = defaultdict(dict)
		business_rating = self.get_business_score(business_id)
		aspects = self.get_business_aspects(business_id)
		for aspect in aspects:
			aspect_summary[aspect] = self.get_aspect_summary(business_id, aspect)

		return dict(business_id=business_id, business_name=" ", \
					business_rating=business_rating, aspect_summary=aspect_summary)

	def generate_model_data(self):
		assert self.user_stars, "please load review data at first"

		data = []
		for review in self.get_all_reviews():
			ave_star = self.user_stars.get(review.user_id)
			if review.stars - ave_star >= 0.5:
				data.append((review.text, 1))
			if review.stars - ave_star <= -0.5:
				data.append((review.text, 0))
			else:
				# drop
				pass

		random.shuffle(data) # 将数据打乱
		train_data, test_data = data[0:len(data) * 0.9], data[len(data) *0.9:]
		return train_data, test_data

	def set_sentiment_model(self, sentiment_model):
		self.sentiment_model = sentiment_model













