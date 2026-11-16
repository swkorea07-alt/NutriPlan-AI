---
title: NutriPlan AI
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# NutriPlan AI

수업 자료 기반 성인 EER + 탄단지 목표를 계산하고 공개 영양 DB 기반으로 식단을 구성하는 교육용 Gradio 웹사이트.

## 실행

```bash
pip install -r requirements.txt
python app.py
```

브라우저에서 `http://localhost:7860`으로 접속. Hugging Face Spaces에 이 폴더의 파일을 업로드하면 별도 Google Drive 연결 없이 실행할 수 있습니다.

- 초기값: 22세, 165cm, 55kg, 비활동적(PA 1.00), 파스타·순두부찌개
- EER 공식 구분은 강의자료와 일치하도록 별도 선택 필드로 유지합니다. 사용자의 성별이 주어지지 않았으므로 기본 화면은 여성용 공식이지만 변경 가능합니다.
- 공개 음식·가공식품 영양 데이터에서 100g 기준의 핵심 4종 값이 모두 제공되는 일부 기록만 추출해 번들로 포함했습니다.
- 출처: 식품의약품안전처 전국통합식품영양성분정보 음식 및 가공식품 공개 자료 (사용자 제공 파일에서 추출).
- 데모는 의료 서비스가 아닙니다.
