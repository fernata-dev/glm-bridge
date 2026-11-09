# glm-bridge

جسر يحوّل شات **glm-ai.chat** لـ **API متوافق مع OpenAI**: `Base URL` + `API Key` + أسماء موديلات.
أي أداة بتقبل "OpenAI-compatible" (n8n، Flowise، Dify، LangChain، LlamaIndex، Continue، مكتبة `openai`…) تقدر تستخدمه من غير ما تعرف إن وراه موقع.

> ⚠️ **اقرا قسم "حدود وصراحة" قبل ما تبني عليه شغل مهم.** الـ API ده مبني على endpoints غير رسمية لموقع طرف ثالث، وممكن تتغير أو تتقفل في أي لحظة.

---

## إيه اللي بيعمله

| الميزة | التفاصيل |
|---|---|
| `POST /v1/chat/completions` | عادي و streaming (SSE)، `stream_options.include_usage`، `stop`، `response_format` (JSON) |
| Tool / Function calling | **محاكاة** عن طريق البرومبت ← بيرجّع `tool_calls` حقيقية بصيغة OpenAI (لازمة لـ AI Agents) |
| Vision | `image_url` (رابط http أو base64) ← بيتحول تلقائياً لموديل الـ vision |
| `POST /v1/images/generations` و `/v1/images/edits` | توليد صور عن طريق `glm-image` |
| `GET /v1/models` | قائمة الموديلات + الأسماء البديلة (aliases) |
| مصادقة | مفاتيح متعددة، `Authorization: Bearer` أو `x-api-key`، مفاتيح ممكن تتخزن كـ hash |
| حماية | rate limit لكل مفتاح، حد لحجم الطلب، حماية SSRF لروابط الصور، طابور تزامن للموقع |
| الاعتمادية | تجديد الـ nonce تلقائياً (refresh واحد حتى لو 50 طلب فشلوا مع بعض)، إعادة محاولة للأخطاء المؤقتة، keep-alive للـ streaming |
| مزوّد احتياطي | لو الموقع وقع أو خلص الحد، الطلب بيتحول لأي API متوافق مع OpenAI (زي API الرسمي لـ Zhipu) |
| تشخيص | `GET /health`، `GET /v1/bridge/status`، وأمر `probe` بيختبر الموقع الحقيقي |

---

## تشغيل سريع

### محلياً

```bash
pip install -r requirements.txt
python -m glm_bridge keygen            # بيطبع مفتاح جديد + الـ hash بتاعه
cp .env.example .env                   # وحط المفتاح في API_KEYS
python -m glm_bridge probe             # ← اعمل ده الأول: بيختبر الموقع الحقيقي وبيقولك الإعدادات المناسبة
python -m glm_bridge serve             # السيرفر على http://localhost:8000
```

### Docker

```bash
cp .env.example .env    # حط API_KEYS
docker compose up -d --build
```

### تجربة سريعة

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"اهلا"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-glm-...")
r = client.chat.completions.create(model="glm-5.3-flash", messages=[{"role": "user", "content": "اهلا"}])
print(r.choices[0].message.content)
```

---

## الربط بأدوات الأتمتة

الإعدادات واحدة في كل الأدوات:

| الحقل | القيمة |
|---|---|
| Base URL | `http://<السيرفر>:8000/v1` |
| API Key | المفتاح اللي ولّدته |
| Model | `glm-5.3-flash` (أو أي اسم من `/v1/models`) |

- **n8n**: Credentials ← *OpenAI* ← حط الـ API Key وغيّر *Base URL*. بعدها استخدم *OpenAI Chat Model* (ينفع مع *AI Agent* لأن الـ tool calling مدعوم). لو الموديل مش ظاهر في القائمة اختار *By ID* واكتبه.
  لو n8n شغال في Docker والجسر على نفس الجهاز: `http://host.docker.internal:8000/v1` (على Linux ضيف `extra_hosts: ["host.docker.internal:host-gateway"]`).
- **Dify**: Settings ← Model Provider ← *OpenAI-API-compatible*.
- **Flowise**: عقدة *ChatOpenAI* ← Additional Parameters ← *BasePath*.
- **LangChain**: `ChatOpenAI(base_url="http://.../v1", api_key="...", model="glm-5.3-flash")`.
- **Make / Zapier** وأي أداة شغالة على السحابة: محتاج رابط **عام HTTPS** (VPS + دومين + Caddy/Nginx، أو Cloudflare Tunnel). ساعتها اظبط `PUBLIC_BASE_URL`.

**أدوات بتفرض اسم موديل زي `gpt-4o`؟** استخدم الأسماء البديلة:
`MODEL_ALIASES={"gpt-4o":"glm-5.3","gpt-4o-mini":"glm-4.5-flash"}`

---

## الموديلات

`glm-5.3` · `glm-5.3-flash` · `glm-5.2` · `glm-5.1` · `glm-4.7` · `glm-4.7-flash` · `glm-4.7-flashx` · `glm-4.6` · `glm-4.5` · `glm-4.5-flash` · `glm-image`

الأسماء العامة بالنقطة (نفس أسماء Zhipu الرسمية، فتشتغل كمان مع المزوّد الاحتياطي). اللي بيتبعت للموقع بالشرطة (`glm-5-3-flash`)، زي السكربت الأصلي.
⚠️ الاسم الوحيد المؤكد من السكربت هو `glm-5-3-flash`. الباقي متوقع بنفس النمط. شغّل `python -m glm_bridge probe --models` عشان تتأكد، وصحّح أي اسم بـ `MODEL_MAP`.
الجسر بيقبل الشكلين من العميل (`glm-5.3-flash` و `glm-5-3-flash`).

---

## إزاي بيتعامل مع قيود الموقع

الموقع بيقبل `{"model", "messages", "image"}` بس، بحد **8000 حرف لآخر رسالة** وصورة واحدة. الجسر بيحوّل أي طلب OpenAI للشكل ده بشكل صريح ومتوقع:

- **رسائل system/developer** + تعليمات الأدوات + تعليمات JSON بتتجمع في بلوك تعليمات واحد، مكانه حسب `SYSTEM_MODE`:
  `auto` (الافتراضي): بيتدمج في آخر رسالة لو فيه مكان، وإلا بيتحط كرسالة تمهيدية في أول المحادثة · `merge` · `primer` · `role` (يتبعت كـ system حقيقي، لو `probe` قال إن الموقع بيقبله).
- **الرسائل الطويلة** (`OVERFLOW_POLICY`): `error` (الافتراضي: يرجع 400 `context_length_exceeded` بدل ما يغيّر كلامك من وراك) · `truncate` (يقص من النص ويسيب البداية والنهاية مع علامة واضحة) · `split` (يقسّم الرسالة لأجزاء، وده محتاج الموقع يكون بيحفظ تاريخ المحادثة).
- **التاريخ الطويل**: الرسائل الأقدم بتتشال لحد ما يدخل في `MAX_CONTEXT_CHARS`.
- **نتايج الأدوات** بتتحول لرسائل user، والرسائل المتتالية من نفس الدور بتتدمج.
- **الصور**: آخر صورة في المحادثة هي اللي بتتبعت. لو الموديل مش vision، الطلب بيتحول لـ `VISION_MODEL` (`VISION_ROUTING`).

كل ده بيظهر في الرد في حقل إضافي `glm_bridge` (الموديل الفعلي، `paid_left`، هل حصل fallback، هل اتقص كلام…) وفي headers زي `X-Bridge-Paid-Left`.

### Tool calling (محاكاة)
الموقع بيرجّع نص بس. الجسر بيوصف الأدوات في البرومبت وبيطلب من الموديل يرد بـ `<tool_call>{json}</tool_call>`، وبعدين بيحوّل الرد لـ `tool_calls` حقيقية (وبيفهم كمان صيغة GLM الأصلية و JSON مباشر). لو الموديل رد بصيغة غلط أو باسم أداة مش موجودة، الرد بيرجع كنص عادي بدل tool call بايظ.
**الصراحة:** ده أضعف من tool calling الأصلي. كويس للـ agents البسيطة (أداة لـ 5 أدوات). لو الـ agent عندك فيه 20 أداة بـ schemas كبيرة هتاكل حد الـ 8000 حرف وهتلاقي أخطاء.

---

## الأخطاء

كل الأخطاء بصيغة OpenAI (`{"error": {"message","type","code"}}`)، فالمكتبات بتفهمها (`RateLimitError`، `AuthenticationError`…).

| الحالة | HTTP | code |
|---|---|---|
| مفتاح ناقص / غلط | 401 | `missing_api_key` / `invalid_api_key` |
| تعديت الـ rate limit بتاع مفتاحك | 429 | `rate_limit_exceeded` |
| الموقع قال `limit` (خلص الحد اليومي) | 429 | `upstream_limit_reached` (+ `upstream_switch` لو اقترح موديل) |
| الموقع `offline` | 503 | `upstream_offline` |
| الموقع رفض الجلسة حتى بعد التجديد (حظر IP أو API اتغير) | 502 | `upstream_auth_failed` |
| الرسالة أطول من الحد | 400 | `context_length_exceeded` |
| الجسر مشغول (طابور التزامن مليان) | 503 | `upstream_busy` |
| موديل مش موجود | 404 | `model_not_found` |

في الـ streaming: أي خطأ بيحصل **قبل أول كلمة** بيرجع بالـ status الصح (مش 200 وبعدين الاتصال يقع)، ده لأن الجسر بيعمل *pre-flight*. لو الخطأ حصل في نص الرد، بيتبعت chunk فيه `error` وبعده `[DONE]`.

---

## المزوّد الاحتياطي

```env
FALLBACK_BASE_URL=https://open.bigmodel.cn/api/paas/v4
FALLBACK_API_KEY=...
FALLBACK_MODEL=glm-4.5-flash
```
لما الموقع يقع أو يخلص حده **قبل ما يرد**، الطلب الأصلي بيتحول زي ما هو (بالأدوات الحقيقية والصور) للمزوّد ده، وبيرجع header `X-Bridge-Provider: fallback`. تأكد من الـ URL والأسعار من الوثائق الرسمية للمزوّد اللي هتختاره.

---

## الأمان

- السيرفر **مش بيشتغل** من غير `API_KEYS`، إلا لو كتبت `AUTH_DISABLED=true` بنفسك.
- خزّن المفاتيح كـ `sha256:...` في `.env` عشان لو الملف اتسرب المفاتيح نفسها متتسربش.
- روابط الصور: بيتم منع العناوين الداخلية (127.0.0.1، 10.x، 169.254.x…) في كل redirect، وبيتأكد إن الملف صورة فعلاً. لو مفاتيحك مع ناس مش واثق فيهم: `ALLOW_REMOTE_IMAGES=false`.
- `LOG_PROMPTS=false` افتراضياً.

---

## حدود وصراحة (اقراها)

1. **ده مش API رسمي.** أي تغيير في الموقع (اسم endpoint، شكل الـ nonce، Cloudflare، حظر IP السيرفر) هيوقف الجسر. علشان كده فيه `probe` و `/v1/bridge/status` و المزوّد الاحتياطي. **أي workflow مهم لازم يكون له fallback.**
2. **الجسر متجرّبش على الموقع الحقيقي وقت التطوير** (الموقع كان محجوب من بيئة التطوير). اتجرّب بمحاكي بنفس البروتوكول اللي في السكربت (`scripts/mock_upstream.py`) وبمكتبة `openai` الرسمية. **شغّل `probe` أول حاجة.**
3. شكل رد `/image` عند الموقع مش معروف. الجسر بيدوّر على الصورة بأي شكل منطقي (رابط، base64، image/*)، بس ممكن يحتاج تعديل.
4. `temperature` و `max_tokens` و `top_p` **بيتقبلوا ويتجاهلوا**، لأن الموقع مش بيستقبلهم. `n>1` مرفوض.
5. أرقام `usage` **تقديرية** (الموقع مش بيرجّع عدد التوكنز).
6. الحد اليومي (`paidLeft`) بتاع الموقع، **والجسر مش بيتحايل عليه**: مفيش تدوير هويات أو كوكيز أو IPs. لما الحد يخلص بترجع 429 بوضوح، والحل إما `FOLLOW_SWITCH` أو المزوّد الاحتياطي.
7. أي حاجة بتبعتها بتعدي على موقع طرف ثالث. **متبعتش أسرار أو بيانات عملاء.**
8. استخدامك للموقع بالشكل ده ممكن يكون ضد شروط استخدامه. دي مسؤوليتك.

---

## التطوير

```bash
pip install -e ".[dev]"
pytest                                   # 80+ اختبار بموقع وهمي (من غير إنترنت)
uvicorn scripts.mock_upstream:app --port 9001   # موقع وهمي لتجربة n8n من غير ما تصرف من الحد
UPSTREAM_BASE_URL=http://127.0.0.1:9001 python -m glm_bridge serve
```

```
glm_bridge/
  upstream.py   عميل الموقع: session/nonce، SSE، طابور التزامن
  convert.py    تحويل طلب OpenAI لشكل الموقع (system، tools، صور، حدود)
  service.py    pre-flight، retries، fallback، تحويل الرد لـ OpenAI/SSE
  tools.py      محاكاة tool calling
  output.py     stop sequences + اكتشاف tool calls أثناء الـ streaming
  auth.py       المفاتيح والـ rate limit
  images.py     التحقق من الصور وحماية SSRF
  imagegen.py   توليد الصور
  app.py        FastAPI routes
```
