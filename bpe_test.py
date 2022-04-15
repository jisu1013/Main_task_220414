from tokenizers import Tokenizer, models

tokenizer = Tokenizer(models.BPE('./output/vocab.json', './output/merges.txt'))

test_str1 = '이스라엘 인 하아레츠의 편집위원실은 트럼프가 “베냐민 네타냐후의 뺨을 후려쳤다”며 미국의 개입은 중동 지역의 게임 규칙을 정하는 데 “러시아에 맞서는 중요한 균형추 역할을 해왔다”고 강조했다.'
test_str2 = '이 단지의 전용면적 별 매매 시세는 59 공급 25평형 13억2000만 15억원 84 공급 34평형 16억5000만 18억5000만원 115 공급 44평형 23억 24억원 135 공급 52평형 25억 28억원 168 공급 62평형 28억5000만원 31억원 198 공급 72평형 30억 32억원 222 공급 81평형 31억 36억원 선에 형성돼 있다.'
test_str3 = '이데일리 정태선 문규영 아주그룹 회장은 4일 “누구도 변화를 예측할 수 없는 시대에 과거의 익숙한 방식에서 벗어나지 못하거나 변화의 속도를 따라가지 못한다면 기업은 결국 도태될 수밖에 없을 것”이라며 “4차 산업혁명과 같은 시대적 요구에 발맞춰 구성원들이 창의력을 발휘할 수 있는 분위기를 조성하고 끊임없이 도전하며 격의 없이 소통할 수 있는 아주 만의 수평적인 기업문화를 구축하는데 앞장서겠다”고 강조했다.'
test_str4 = '한국갤럽이 지난 13 15일 전국 성인 1천3명을 대상으로 실시한 여론조사 95 신뢰 수준, 표본오차 3.1%포인트 에 따르면 문 대통령의 직무수행에 대한 긍정 평가는 83 로 1주 전보다 1 포인트 상승했다.'

enc_1 = tokenizer.encode(test_str1, is_pretokenized=False)
enc_2 = tokenizer.encode(test_str2, is_pretokenized=False)
enc_3 = tokenizer.encode(test_str3, is_pretokenized=False)
enc_4 = tokenizer.encode(test_str4, is_pretokenized=False)

print(enc_1.word_ids)
print(enc_1.tokens)
print(enc_1.ids)
print(len(enc_1.ids))

print(enc_2.word_ids)
print(enc_2.tokens)
print(enc_2.ids)
print(len(enc_2.ids))

print(enc_3.word_ids)
print(enc_3.tokens)
print(enc_3.ids)
print(len(enc_3.ids))

print(enc_4.word_ids)
print(enc_4.tokens)
print(enc_4.ids)
print(len(enc_4.ids))