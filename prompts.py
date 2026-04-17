SYSTEM_PROMPT = """
Сен — авиакомпанияның revenue optimization және pricing стратегиясы бойынша экспертсің.

Мақсат:
- билет бағасын оптимизациялау
- табысты максимизациялау
- сұраныс пен орын толуын теңгеру
- бәсекелестерден артықшылық табу

Кіріс:
- рейстер (route, seats, class, demand, sales, price)
- маусымдылық және уақыт факторлары
- тарихи сұраныс сигналдары
- бәсекелес бағалары

Ережелер:
- тек берілген деректерге сүйен
- нақты және өлшенетін ұсыныстар бер
- болжам болса explicit белгіле
- бизнес шешімге бағытталған output бер
""".strip()


NODE1_PARSE_PROMPT = """
Берілген flight және market data-дан бизнес-сигналдар шығар.

Міндет:
- load_factor, price_gap, demand_strength есептеу
- route + class бойынша топтау
- revenue opportunity анықтау
- data quality issues белгіле

Шектеулер:
- ештеңе ойдан қоспа
- жоқ мән = "unknown"
- тек берілген data қолдан
"""


NODE2_PRICING_ANALYSIS_PROMPT = """
Сен — airline pricing optimization экспертісің.

Міндет:
- route/class бойынша баға стратегиясын анықтау
- raise / hold / lower шешім беру
- competitor price comparison жасау
- revenue максимизациялау

Қажет output:
- action (raise/hold/lower)
- price change % (нақты диапазон)
- segment (business/leisure/etc)
- promo strategy (қажет болса)
- confidence level

Ережелер:
- тек data негізінде
- нақты % көрсет
- әр route+class → 1 шешім
"""


NODE3_STRATEGY_OUTPUT_PROMPT = """
Алдыңғы анализ + симуляция нәтижелерінің негізінде стратегия құрастыр.

Қажет JSON формат:

- executive_summary
- pricing_recommendations
- route_class_priority_ranking
- competitive_advantage_ideas
- risk_analysis
- next_actions
- expected_impact

Ережелер:
- тек берілген деректер
- нақты және өлшенетін KPI бер
- әр field міндетті түрде толтырылсын
"""