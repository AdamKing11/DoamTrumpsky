

class NPhoner:


	def __init__(self,text_file = 'cleaned_all.txt', max_nphone = 1, load = False):
		if not load:
			self.nphones = self.build_nphones(text_file, max_nphone)
		else:
			self.nphones = self.load(text_file)


	def build_nphones(self, text_file, max_nphone):
		phones = {}
		
		with open(text_file, 'r', max_nphone) as rf:
			for l in rf:
				l = l.rstrip()
				for i in range(len(l)):
					for j in range(max_nphone):
						n = ''.join(l[i:i+j+1])
						if n in phones:
							phones[n] += 1
						else:
							phones[n] = 1

		phones = self.cull_nphones(phones)

		return phones

	def cull_nphones(self, nphones, min_cnt = 10, med_mod = .5):

		nphones = dict((n,nphones[n]) for n in nphones if nphones[n]>min_cnt)

		uniphones = len([n for n in nphones if (len(n)==1 and n != ' ')])
		uniphone_median = sum([nphones[n] for n in nphones if (len(n) == 1  and n != ' ')])/uniphones
		uniphone_median *= med_mod

		nphones = dict((n,nphones[n]) for n in nphones if (len(n) == 1 or nphones[n]>uniphone_median))

		return nphones

	def save(self, save_file):
		with open(save_file, 'w') as wf:
			for n in sorted(self.nphones, key=self.nphones.get, reverse=True):
				wf.write(n + '\t' + str(self.nphones[n]) + '\n')

	def load(self, load_file):
		phones = {}
		with open(load_file, 'r') as rf:
			for l in rf:
				l = l.rstrip().rsplit('\t')
				phones[l[0]] = int(l[1])
		return phones


	def get_nphones(self):
		return self.nphones