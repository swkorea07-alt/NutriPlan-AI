"""NutriPlan AI - daily meal planner, derived from the user's v3 Colab project.
This educational demo is not medical or allergy advice. Data: publicly released Korean food nutrition datasets.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
from scipy.optimize import lsq_linear
import gradio as gr
DATA_DIR = Path(__file__).resolve().parent / "data"
COLS = ['식품코드','식품명','영양성분함량기준량','에너지(kcal)',
        '탄수화물(g)','단백질(g)','지방(g)','나트륨(mg)','식품대분류명','당류(g)']
MACROS = ['에너지(kcal)','탄수화물(g)','단백질(g)','지방(g)']

def load_nutrition(path, source):
    if not path.exists():
        raise FileNotFoundError(f'파일이 없습니다: {path.name} (Google Drive data 폴더를 확인하세요)')
    last_error = None
    for enc in ('utf-8-sig','cp949','utf-8'):
        try:
            df = pd.read_csv(path, encoding=enc, usecols=lambda x: x in COLS,
                             low_memory=False, dtype={'식품코드':'string','식품명':'string'})
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error
    required = ['식품명','영양성분함량기준량']+MACROS
    missing = set(required)-set(df.columns)
    if missing:
        raise ValueError(f'{path.name}에 필수 컬럼이 없습니다: {sorted(missing)}')
    n_original = len(df)
    df = df.copy()
    df['기준량원문'] = df['영양성분함량기준량'].astype(str).str.strip()
    df = df[df['기준량원문'].str.fullmatch(r'100\s*g', case=False, na=False)].copy()
    for col in MACROS+['나트륨(mg)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','',regex=False), errors='coerce')
    df = df.dropna(subset=MACROS)
    df = df[(df[MACROS]>=0).all(axis=1) &
            (df['에너지(kcal)'].between(15,900)) &
            (df['탄수화물(g)']<=100) & (df['단백질(g)']<=100) & (df['지방(g)']<=100)].copy()
    df['식품명'] = df['식품명'].astype(str).str.strip()
    df = df[df['식품명'].str.len().between(1,90)]
    df['출처유형'] = source
    df = df.drop_duplicates(subset=['식품명']+MACROS)
    df = df.reset_index(drop=True)
    print(f'{source}: 원본 {n_original:,}건 → 100g·필수 영양정보 완비 {len(df):,}건')
    return df


food_df = load_nutrition(DATA_DIR/'food.csv', '일반 음식')
processed_df = load_nutrition(DATA_DIR/'processed_food.csv', '가공식품')

PA_MAP = {'비활동적 (1.00)':1.00, '가벼운 활동 (1.11)':1.11,
          '보통 활동 (1.25)':1.25, '매우 활동적 (1.48)':1.48}
RATIOS = {'carb':0.65, 'protein':0.15, 'fat':0.20}

def estimate_needs(age, height_cm, weight_kg, formula, activity):
    age, height_cm, weight_kg = float(age),float(height_cm),float(weight_kg)
    if not (19 <= age <= 100):
        raise ValueError('현재 버전은 성인 19~100세만 사용할 수 있습니다.')
    if not (120 <= height_cm <= 230 and 30 <= weight_kg <= 250):
        raise ValueError('키 또는 몸무게가 입력 범위를 벗어났습니다.')
    pa = PA_MAP[activity]
    h = height_cm / 100
    if formula=='남성용 공식':
        kcal = 662 - 9.53*age + pa*(15.91*weight_kg+539.6*h)
    else:
        kcal = 354 - 6.91*age + pa*(9.36*weight_kg+726*h)
    if not (900 <= kcal <= 5000):
        raise ValueError('추정 열량이 예상 범위를 벗어납니다. 입력값을 확인하세요.')
    return dict(kcal=float(kcal), carb_g=kcal*.65/4,
                protein_g=kcal*.15/4, fat_g=kcal*.20/9)


RULES = {
 '밥': r'^(?:쌀밥|흰쌀밥|현미밥|잡곡밥|보리밥|흑미밥|오곡밥|콩밥|귀리밥)(?:_|$)',
 '단백질 반찬': r'^(?:닭가슴살|닭구이|고등어구이|연어구이|갈치구이|계란말이|달걀말이|계란찜|달걀찜|두부구이|두부조림|불고기|돼지고기수육|제육볶음|소고기구이|삼치구이|생선구이|갈비구이)(?:_|$)',
 '채소 반찬': r'^(?:시금치나물|콩나물무침|숙주나물|상추겉절이|오이무침|브로콜리|애호박볶음|버섯볶음|가지나물|배추김치|깍두기|샐러드)(?:_|$)',
 '국': r'^(?:미역국|된장국|콩나물국|북엇국|무국|된장찌개|김치찌개)(?:_|$)',
}
# 아침 전용 후보군: 점심/저녁의 고기·갈비·매운 찌개 등이 들어오지 않게 별도 구성
# 아침은 죽/담백한 밥/일부 식빵 + 달걀 + 가벼운 채소로만 구성합니다.
BREAKFAST_RULES = {
    '아침 주식': r'^(?:쌀밥|현미밥|잡곡밥|팥죽|호박죽|전복죽|닭죽|식빵|식빵_통밀식빵|식빵_통밀 씨드 식빵)$',
    '아침 단백질': r'^(?:달걀찜(?:_|$)|달걀말이(?:_|$))',
    '아침 채소': r'^(?:샐러드_(?:새싹|양배추|양상추|채소)$|시금치나물$|콩나물무침$|오이무침_부추$)'
}
BREAKFAST_POOLS = {}
for category, pattern in BREAKFAST_RULES.items():
    subset = food_df[food_df['식품명'].str.contains(pattern, regex=True, na=False)].copy()
    BREAKFAST_POOLS[category] = subset.drop_duplicates(subset='식품명').reset_index(drop=True)
    print(category, ':', len(BREAKFAST_POOLS[category]), '개')
assert all(len(x) > 0 for x in BREAKFAST_POOLS.values()), '아침용 음식 후보가 부족합니다. 음식 CSV를 확인하세요.'

POOLS = {}
for group, pat in RULES.items():
    pool = food_df[food_df['식품명'].str.contains(pat, regex=True, na=False)].copy()
    pool = pool[~pool['식품명'].str.contains(r'(?:간편조리세트|피자|치즈버스트|더블치즈)',na=False,regex=True)]
    POOLS[group] = pool.drop_duplicates(subset='식품명').reset_index(drop=True)
    print(group, ':', len(POOLS[group]), '개')

SNACK_PAT = r'^(?:두유|우유|그릭요거트|플레인요거트|요거트|요구르트)(?:_|\s|$)'
snack_pool = processed_df[processed_df['식품명'].str.contains(SNACK_PAT,regex=True,na=False)].copy()
snack_pool = snack_pool[(snack_pool['에너지(kcal)'].between(30,220)) &
                        (snack_pool['당류(g)'].fillna(999)<30 if '당류(g)' in snack_pool.columns else True)]
snack_pool = snack_pool.drop_duplicates(subset='식품명').reset_index(drop=True)
print('가공식품 간식 후보:',len(snack_pool))
assert min(len(POOLS[g]) for g in POOLS)>=3, '식사 후보가 부족합니다. 음식 CSV를 확인하세요.'

# 실제 음식 CSV에서 영양성분이 완비된 식품만 선호 메뉴 검색에 노출
# 유사명칭이 많은 경우에는 일반 음식 데이터의 첫 기록을 사용합니다.
FAVORITE_DF = food_df.drop_duplicates(subset='식품명').copy()
FAVORITE_DF = FAVORITE_DF[~FAVORITE_DF['식품명'].str.contains(
    r'(?:간편조리세트|피자_.*파스타|치즈버스트)', regex=True, na=False)]
FAVORITE_DF = FAVORITE_DF.set_index('식품명', drop=False)
FAVORITE_NAMES = sorted(FAVORITE_DF.index.astype(str).tolist())

BOUNDS = {'아침 주식':(90,180),'아침 단백질':(55,115),'아침 채소':(40,90),
          '밥':(110,340), '단백질 반찬':(70,230),
          '채소 반찬':(50,170),'국':(100,240),
          '선호 메인':(140,400),'선호 국/찌개':(160,350),'간식':(80,220)}
SECTIONS = [('아침','아침 주식'),('아침','아침 단백질'),('아침','아침 채소'),
            ('점심','밥'),('점심','단백질 반찬'),('점심','채소 반찬'),('점심','국'),
            ('저녁','밥'),('저녁','단백질 반찬'),('저녁','채소 반찬'),('저녁','국')]

def match_favorites(picked, typed, avoided=''):
    """선택한 정확한 메뉴명 + 쉼표로 입력한 음식 키워드를 실제 DB 메뉴와 연결."""
    wishes = list(picked or []) + [x.strip() for x in re.split(r'[,，、\n]+',typed or '') if x.strip()]
    found, missing, used = [], [], set()
    for wish in wishes[:12]:
        if wish in FAVORITE_DF.index:
            name = wish
        else:
            # '파스타' 입력 → 실제 파스타 메뉴 검색. 피자 등 관련없는 상품 제외
            matches = FAVORITE_DF[FAVORITE_DF['식품명'].str.contains(re.escape(wish),case=False,na=False)]
            if wish.strip() == '파스타':
                matches = FAVORITE_DF[FAVORITE_DF['식품명'].str.contains(r'스파게티|파스타',case=False,regex=True,na=False)]
            if wish.strip() == '파스타':
                matches=matches[matches['식품명'].str.contains('스파게티|파스타',na=False)]
                matches=matches[~matches['식품명'].str.contains('피자',na=False)]
            if matches.empty:
                missing.append(wish)
                continue
            # '순두부찌개'같은 메뉴는 '순두부찌개_김치' 등의 구체적인 기록으로 연결
            matches = matches.assign(_priority=matches['식품명'].str.len())
            name = ('스파게티_토마토소스' if wish.strip() == '파스타' and '스파게티_토마토소스' in matches['식품명'].values else matches.sort_values('_priority').iloc[0]['식품명'])
        if avoided and avoided.casefold() in name.casefold():
            missing.append(wish+' (제외 키워드와 충돌)')
            continue
        if name not in used:
            found.append(name); used.add(name)
    return found, missing

def _favorite_group(name):
    # 찌개·국은 밥과 함께, 면·파스타류는 밥을 대체하는 메인 요리로 취급
    if re.search(r'(찌개|국|탕)(?:_|\s|$)',name):
        return '선호 국/찌개'
    return '선호 메인'

def pick_foods(rng, avoided='', favorites=()):
    """점심/저녁에 선호 음식을 우선 확정한 뒤 다른 식품을 채워 식단을 구성."""
    selected=[]; used=set()
    schedules={'아침':['아침 주식','아침 단백질','아침 채소'],
               '점심':['밥','단백질 반찬','채소 반찬','국'],
               '저녁':['밥','단백질 반찬','채소 반찬','국']}
    # 한 끼에 선호 음식 1개, 기본적으로 점심/저녁에 배치
    assignments = {}
    for meal, name in zip(['점심','저녁'],favorites[:2]):
        group = _favorite_group(name)
        assignments[meal] = (name, group)
        if group == '선호 국/찌개':
            schedules[meal].remove('국')
        else:
            schedules[meal].remove('밥')
        # 면·파스타 식단과 국/찌개 중복을 피함
        if group == '선호 메인':
            schedules[meal].remove('국')
    for meal in ('아침','점심','저녁'):
        if meal in assignments:
            name, group = assignments[meal]
            food=FAVORITE_DF.loc[name]
            selected.append({'meal':meal,'group':group,'food':food,'favorite':True})
            used.add(name)
        for group in schedules[meal]:
            pool=BREAKFAST_POOLS[group] if group in BREAKFAST_POOLS else POOLS[group]
            if avoided:
                pool=pool[~pool['식품명'].str.contains(re.escape(avoided),case=False,na=False)]
            pool=pool[~pool['식품명'].isin(used)]
            if pool.empty:
                raise ValueError(f'{group} 후보가 없습니다. 제외 키워드를 바꿔주세요.')
            row=pool.iloc[int(rng.integers(0,len(pool)))]
            selected.append({'meal':meal,'group':group,'food':row,'favorite':False})
            used.add(row['식품명'])
    if len(snack_pool):
        pool=snack_pool
        if avoided:
            pool=pool[~pool['식품명'].str.contains(re.escape(avoided),case=False,na=False)]
        if not pool.empty:
            row=pool.iloc[int(rng.integers(0,len(pool)))]
            selected.append({'meal':'간식','group':'간식','food':row,'favorite':False})
    return selected

def fit_portions(items, needs):
    a=np.array([[float(t['food'][c])/100 for t in items] for c in MACROS],float)
    n=len(items)
    targets=np.array([needs['kcal'],needs['carb_g'],needs['protein_g'],needs['fat_g']])
    main=a/np.maximum(targets[:,None],1)
    mat=list(main*np.array([4.0,2.0,2.0,1.6])[:,None])
    goal=[4,2,2,1.6]
    meal_ratios={'아침':.22,'점심':.38,'저녁':.34,'간식':.06}
    for meal,ratio in meal_ratios.items():
        ix=np.array([t['meal']==meal for t in items],float)
        if not ix.any():continue
        mat.append(ix*a[0]/needs['kcal']*4.0)
        goal.append(ratio*4.0)
    lo=np.array([BOUNDS[t['group']][0] for t in items],float)
    hi=np.array([BOUNDS[t['group']][1] for t in items],float)
    center=(lo+hi)/2
    reg=np.eye(n)*(.14/np.maximum(center,1))
    mat.extend(reg)
    goal.extend(.14*np.ones(n))
    res=lsq_linear(np.array(mat),np.array(goal),bounds=(lo,hi),max_iter=150)
    grams=np.clip(np.round(res.x/5)*5,lo,hi)
    totals=a@grams
    rel=np.abs(totals-targets)/np.maximum(targets,1)
    score=float(rel@np.array([5,1.5,1.5,1]))
    return score,grams,totals

def generate_menu(needs, avoided='', favorites=(), seed=None, n_attempts=90):
    rng=np.random.default_rng(seed)
    best=None
    for _ in range(n_attempts):
        items=pick_foods(rng,avoided.strip(),favorites)
        score,grams,totals=fit_portions(items,needs)
        if best is None or score<best[0]:best=(score,items,grams,totals)
    return best


def nutrition_ui(age, formula, height, weight, activity, favorite_selected, favorite_typed, exclude, seed_text):
    try:
        needs=estimate_needs(age,height,weight,formula,activity)
        seed=int(seed_text) if str(seed_text).strip() else None
        favorites, missing=match_favorites(favorite_selected,favorite_typed,exclude or '')
        score,items,grams,totals=generate_menu(needs,exclude or '',favorites=favorites,seed=seed)
        rows=[]
        for entry,g in zip(items,grams):
            f=entry['food']; factor=g/100
            rows.append({'끼니':entry['meal'],'구성':entry['group'],
                         '선호 반영':'★ 좋아하는 음식' if entry.get('favorite') else '',
                         '음식명':str(f['식품명']),'섭취량(g)':int(g),
                         '열량(kcal)':round(float(f['에너지(kcal)'])*factor,1),
                         '탄수화물(g)':round(float(f['탄수화물(g)'])*factor,1),
                         '단백질(g)':round(float(f['단백질(g)'])*factor,1),
                         '지방(g)':round(float(f['지방(g)'])*factor,1),
                         '데이터 출처':str(f['출처유형'])})
        table=pd.DataFrame(rows)
        target=np.array([needs['kcal'],needs['carb_g'],needs['protein_g'],needs['fat_g']])
        comp=pd.DataFrame({'항목':['열량(kcal)','탄수화물(g)','단백질(g)','지방(g)'],
                           '목표':[round(v,1) for v in target],
                           '식단 합계':[round(v,1) for v in totals],
                           '목표 대비(%)':[round(100*a/t,1) for a,t in zip(totals,target)]})
        macro_cal=np.array([totals[1]*4,totals[2]*4,totals[3]*9])
        pct=macro_cal/macro_cal.sum()*100 if macro_cal.sum()>0 else [0,0,0]
        included=[str(x['food']['식품명']) for x in items if x.get('favorite')]
        favorite_info=('\n- **식단에 실제 반영된 선호 음식:** '+', '.join(included)) if included else '\n- **선호 음식:** 지정하지 않음'
        if len(favorites)>2:
            favorite_info+='\n- 선호 음식은 최대 2개까지 하루 식단에 우선 반영해요. 나머지는 다음 식단 생성 때 선택해 주세요.'
        if missing:
            favorite_info+='\n- **매칭되지 않은 입력:** '+', '.join(missing)+' (다른 키워드로 검색해 보세요.)'
        info=(f'### 하루 추정 에너지 필요량: **{needs["kcal"]:,.0f} kcal**\n'
              f'- 강의자료 배분 목표: 탄수화물 **{needs["carb_g"]:.1f}g** (65%), '
              f'단백질 **{needs["protein_g"]:.1f}g** (15%), 지방 **{needs["fat_g"]:.1f}g** (20%)\n'
              f'- 추천 식단 총열량: **{totals[0]:,.0f} kcal** '
              f'(목표 대비 {totals[0]/needs["kcal"]*100:.1f}%)\n'
              f'- 식단 탄단지 에너지 비율(4/4/9 환산): '
              f'**{pct[0]:.1f}% / {pct[1]:.1f}% / {pct[2]:.1f}%**\n'
              +favorite_info+'\n\n'
              '⚠️ 이 식단은 영양수치 기반 자동 조합으로, 개별 알레르기·질환·임신/수유·식품 가격을 반영하지 않습니다. '
              '선호 음식의 포함 여부를 우선하며 모든 영양 목표를 정확히 충족한다고 보장하지 않습니다. '
              '음식의 조리법과 실제 섭취량에 따라 수치가 달라질 수 있습니다. '
              '아침은 담백한 메뉴로 제한하며 열량 비중은 최적화 목표치이므로 정확히 일치하지 않을 수 있습니다.')
        return info,table,comp
    except Exception as e:
        return '⚠️ '+str(e),pd.DataFrame(),pd.DataFrame()


css = '\n:root {--body-background-fill:#f8f5ef; --block-background-fill:#fffdf9; --primary-500:#60866d;--primary-600:#42684f;}\n.gradio-container {max-width:1190px !important; margin: auto !important;}\n#hero {background:linear-gradient(110deg,#f0eee2,#e1ece0);border-radius:22px;padding:26px 30px;margin:20px 0 18px;}\n#hero h1{font-size:31px;letter-spacing:-.8px;color:#304d39;margin:0;}\n#hero p{color:#526657;margin:8px 0 0;}\n#sidebar{background:#fffdf9;border:1px solid #e8e5dc;border-radius:18px;padding:19px;}\n#results{background:#fffdf9;border:1px solid #e8e5dc;border-radius:18px;padding:19px;}\n'

with gr.Blocks(title='NutriPlan AI · 하루 맞춤 식단') as demo:
    gr.HTML('<div id="hero"><h1>🌿 NutriPlan AI</h1><p>내 생활과 취향에 맞춘, 현실적인 하루 식단. 아침은 가볍게 · 점심은 든든하게 · 저녁은 균형 있게.</p></div>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_id='sidebar'):
            gr.Markdown('### 01 · 나의 기본 정보')
            with gr.Row():
                age=gr.Number(value=22,label='나이 (세)',precision=0)
                formula=gr.Radio(['여성용 공식','남성용 공식'],value='여성용 공식',label='EER 공식')
            with gr.Row():
                height=gr.Number(value=165,label='키 (cm)')
                weight=gr.Number(value=55,label='체중 (kg)')
            activity=gr.Dropdown(list(PA_MAP),value='비활동적 (1.00)',label='평소 활동량',info='low = 비활동적 (PA 1.00)')
            gr.Markdown('### 02 · 좋아하는 음식')
            favorites=gr.Dropdown(choices=FAVORITE_NAMES,multiselect=True,filterable=True,label='DB에서 음식 검색해서 선택',info='음식 이름을 검색해 여러 개 선택할 수 있어요.')
            favorite_text=gr.Textbox(value='파스타, 순두부찌개',lines=2,label='좋아하는 음식 직접 입력',info='쉼표로 구분하세요. 비슷한 이름의 실제 음식 데이터를 찾아 반영합니다.')
            exclude=gr.Textbox(label='제외할 음식 키워드 (선택)',placeholder='예: 고등어')
            seed=gr.Textbox(label='추천 재현 번호 (선택)',placeholder='예: 42')
            go=gr.Button('🍽️ 나에게 맞는 식단 만들기',variant='primary',size='lg')
        with gr.Column(scale=6, elem_id='results'):
            gr.Markdown('### 03 · 나의 영양 목표와 식단')
            result=gr.Markdown('왼쪽의 입력값을 확인하고 **식단 만들기**를 눌러주세요.')
            comparison=gr.Dataframe(label='목표와 추천 식단 비교',interactive=False,wrap=True)
    menu=gr.Dataframe(label='🥗 아침 · 점심 · 저녁 · 간식 식단 (★ 선호 음식 반영)',interactive=False,wrap=True)
    gr.Markdown("""<small>**계산 기준:** 제공된 수업 자료의 19세 이상 성인 EER 공식과 탄수화물 65% · 단백질 15% · 지방 20% 배분. 끼니별 열량 목표는 아침 22%, 점심 38%, 저녁 34%, 간식 6%입니다. 실제 메뉴와 영양소의 차이는 발생할 수 있습니다.</small>

<small>⚠️ **주의:** 교육용 시제품입니다. 의학적 처방, 알레르기 안전성 판정, 임신·수유 중 영양 계획 및 질환별 식단에 사용하지 마세요. 실제 영양성분은 조리법과 섭취량에 따라 달라집니다. 입력 정보는 이 데모에서 별도 계정에 저장하지 않도록 구성했지만, 배포 서비스의 서버 운영 환경은 별도 점검해야 합니다.</small>""")
    go.click(nutrition_ui,inputs=[age,formula,height,weight,activity,favorites,favorite_text,exclude,seed],outputs=[result,menu,comparison])

if __name__=='__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, css=css, theme=gr.themes.Soft(primary_hue='green', neutral_hue='stone'))
