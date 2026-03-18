from langchain_core.runnables import Runnable
from typing import Any, Iterator, AsyncIterator


class MyCustomRunnable(Runnable):
    def invoke(self, input: Any, config: dict = None) -> Any:
        """同步调用 - 核心方法"""
        # chain.invoke
        pass

    def stream(self, input: Any, config: dict = None) -> Iterator[Any]:
        """同步流式调用"""
        # chain.stream流式调用
        pass

    def batch(self, inputs: list, config: dict = None) -> list:
        """批量处理"""
        # chain.batch() - 批量处理
        # 1、需要真正的逐字流式 → 使用 stream() 或异步并发
        # 2、需要最高性能的批量处理 → 使用 batch() + streaming=False
        pass

    def ainvoke(self, input: Any, config: dict = None) -> Any:
        """异步调用"""
        # 异步调用 chain.ainvoke()
        pass

    def astream(self, input: Any, config: dict = None) -> AsyncIterator[Any]:
        """异步流式调用"""
        # 异步流式调用 chain.astream()
        pass

    def abatch(self, inputs: list, config: dict = None) -> list:
        """异步批量处理"""
        # chain.abatch()
        pass


# 内置Runnable函数：1、RunnableLambda - 函数包装器
# 适用于数据预处理、复杂业务逻辑
from langchain_core.runnables import RunnableLambda

# Example 1：复杂业务逻辑处理
# 电商订单处理
validate_order = RunnableLambda(lambda order: order if order["amount"] > 0 else None)
apply_discount = RunnableLambda(lambda order: {
    **order,
    "final_amount": order["amount"] * 0.9 if order.get("is_vip") else order["amount"]
})
generate_receipt = RunnableLambda(lambda order: f"""
    订单号: {order['id']}
    原金额: ${order['amount']}
    最终金额: ${order['final_amount']}
    感谢您的购买！
    """)

# 连接多个处理步骤，每个步骤的输出作为下一个步骤的输入
order_processing_chain = validate_order | apply_discount | generate_receipt

# 处理订单
order = {"id": "12345", "amount": 100, "is_vip": True}
receipt = order_processing_chain.invoke(order)
print(receipt)

# 内置Runnable函数：2、RunnableParallel - 并行处理、字段映射
# 1、并行：同时执行多个独立任务，然后将结果合并。适用于需要从不同来源获取数据或并行处理不同计算的场景。
# 2、映射，用于对输入数据的特定字段进行转换或提取，生成新的数据结构。适用于需要从复杂输入中提取、转换特定字段的场景。

# Example 1：
# 用户画像分析系统，电商平台需要为用户生成完整的画像，包括：
# 1、基本信息
# 2、订单历史分析
# 3、浏览行为分析
# 4、社交关系分析
# 这些数据来自不同的服务，可以并行获取。
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time


# 模拟不同数据源的获取函数
def fetch_user_basic_info(user_id: str):
    """模拟获取用户基本信息（来自用户服务）"""
    print(f"🔍 获取用户 {user_id} 的基本信息...")
    time.sleep(0.5)  # 模拟网络延迟
    return {
        "user_id": user_id,
        "name": "张三",
        "age": 28,
        "membership_level": "VIP",
        "registration_date": "2022-01-15"
    }


def fetch_order_history(user_id: str):
    """模拟获取订单历史（来自订单服务）"""
    print(f"🛒 获取用户 {user_id} 的订单历史...")
    time.sleep(0.8)  # 模拟较慢的服务
    return {
        "total_orders": 45,
        "total_spent": 12800.50,
        "avg_order_value": 284.46,
        "favorite_category": "电子产品",
        "last_order_date": "2024-01-20"
    }


def fetch_browsing_behavior(user_id: str):
    """模拟获取浏览行为（来自行为分析服务）"""
    print(f"📊 获取用户 {user_id} 的浏览行为...")
    time.sleep(0.3)
    return {
        "weekly_visits": 12,
        "avg_session_duration": "8分30秒",
        "most_browsed_category": "家居用品",
        "cart_abandonment_rate": "15%"
    }


def fetch_social_connections(user_id: str):
    """模拟获取社交关系（来自社交服务）"""
    print(f"👥 获取用户 {user_id} 的社交关系...")
    time.sleep(0.6)
    return {
        "followers_count": 156,
        "following_count": 89,
        "reviews_written": 23,
        "helpful_votes": 45
    }

# 创建并行处理链
# 注意：RunnableParallel内的任务不支持相互依赖，如某个任务依赖前置任务的输出
# 需改用RunnableSequence按顺序执行 或 RunnablePassthrough方法传输数据
user_profile_chain = RunnableParallel({
    "basic_info": RunnableLambda(lambda x: fetch_user_basic_info(x["user_id"])),
    "order_history": RunnableLambda(lambda x: fetch_order_history(x["user_id"])),
    "browsing_behavior": RunnableLambda(lambda x: fetch_browsing_behavior(x["user_id"])),
    "social_connections": RunnableLambda(lambda x: fetch_social_connections(x["user_id"]))
})

print(user_profile_chain)

# 创建分析总结链
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

analysis_prompt = ChatPromptTemplate.from_template("""
基于以下用户数据，生成一份完整的用户画像分析：

用户基本信息:
{basic_info}

订单行为:
{order_history}

浏览习惯:
{browsing_behavior}

社交影响力:
{social_connections}

请从以下角度进行分析：
1. 用户价值评估
2. 购买偏好分析  
3. 潜在需求识别
4. 个性化推荐建议

用中文回复，结构清晰。
""")

# 完整处理管道:管道是顺序执行，不是同步
# # 当 llm 返回时，实际上返回的是 AIMessage 对象
# # AIMessage(content='模型生成的文本内容', additional_kwargs={}, ...)
# # x.content返回的是AIMessage的纯文本
full_analysis_chain = user_profile_chain | analysis_prompt | llm | RunnableLambda(lambda x: x.content)

# 执行分析
print("🚀 开始生成用户画像分析...")
start_time = time.time()

user_id = "U123456789"
result = user_profile_chain.invoke({"user_id": user_id})
# 完整结果改成full_analysis_chain
# result = full_analysis_chain.invoke({"user_id": user_id})

end_time = time.time()
print(f"\n⏱️ 总处理时间: {end_time - start_time:.2f}秒")
print("\n" + "="*60)
print("📋 用户画像分析报告:")
print("="*60)
print(result)

# Example 2：用户数据标准化处理
# 电商系统从不同渠道获取用户数据，格式不统一，需要：
# 1、提取关键字段
# 2、标准化格式
# 3、计算衍生字段
# 4、过滤敏感信息

from langchain_core.runnables import RunnableLambda, RunnableParallel
from datetime import datetime, date
import re


def create_user_data_mapper():
    """创建用户数据映射处理器"""

    # RunnableParallel 接收一个字典，其中：
    # 键（Key）：输出字段的名称
    # 值（Value）：处理该字段的 Runnable 对象

    # 1、并行执行所有字段的处理函数
    # 2、收集结果并组装成指定结构的输出字典
    # 3、自动过滤掉未添加的字段（如密码、SSN等敏感信息）

    # 错误隔离：
    # 单个字段处理失败不影响其他字段
    # 数据缺失了某个字段，函数中可编写每个字段的对应处理

    user_data_mapper = RunnableParallel({
        # 提取和标准化基本字段
        "user_id": RunnableLambda(lambda x: str(x.get("id", ""))),
        "full_name": RunnableLambda(
            lambda x: f"{x.get('first_name', '').strip()} {x.get('last_name', '').strip()}".strip()),

        # 标准化联系信息
        "email": RunnableLambda(lambda x: x.get("email", "").lower().strip()),
        "phone": RunnableLambda(lambda x: re.sub(r'\D', '', x.get("phone", ""))),  # 只保留数字

        # 计算衍生字段
        "age": RunnableLambda(lambda x: calculate_age(x.get("birth_date"))),
        "age_group": RunnableLambda(lambda x: get_age_group(calculate_age(x.get("birth_date")))),

        # 地址标准化
        "address": RunnableLambda(lambda x: {
            "street": x.get("street", "").title(),
            "city": x.get("city", "").title(),
            "state": x.get("state", "").upper(),
            "zip_code": x.get("zip_code", ""),
            "country": "US"  # 默认值
        }),

        # 会员状态处理
        "membership": RunnableLambda(lambda x: {
            "tier": x.get("membership_tier", "basic").lower(),
            "since": x.get("join_date", ""),
            "status": "active" if x.get("is_active", False) else "inactive"
        }),

        # 行为评分
        "behavior_score": RunnableLambda(lambda x: calculate_behavior_score(x)),

        # 过滤掉的字段（不包含在输出中）
        # 比如：password, ssn, raw_birth_date 等敏感信息
    })

    return user_data_mapper


# 辅助函数
def calculate_age(birth_date_str):
    """计算年龄"""
    if not birth_date_str:
        return None
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except:
        return None


def get_age_group(age):
    """获取年龄分组"""
    if age is None:
        return "unknown"
    elif age < 18:
        return "teen"
    elif 18 <= age < 25:
        return "young_adult"
    elif 25 <= age < 35:
        return "adult"
    elif 35 <= age < 50:
        return "middle_aged"
    else:
        return "senior"


def calculate_behavior_score(user_data):
    """计算用户行为评分"""
    score = 50  # 基础分

    # 基于各种因素调整分数
    if user_data.get("email_verified", False):
        score += 10
    if user_data.get("phone_verified", False):
        score += 10
    if user_data.get("has_orders", False):
        score += 15
    if user_data.get("account_age_days", 0) > 365:
        score += 10

    return min(score, 100)  # 不超过100


# 使用用户数据映射器
print("👤 用户数据标准化处理")
print("=" * 50)

user_mapper = create_user_data_mapper()

# 测试不同来源的用户数据
test_users = [
    {
        "id": 12345,
        "first_name": " john ",
        "last_name": " DOE ",
        "email": " John.Doe@EXAMPLE.COM ",
        "phone": "(555) 123-4567",
        "birth_date": "1990-05-15",
        "street": "123 main st",
        "city": "new york",
        "state": "ny",
        "zip_code": "10001",
        "membership_tier": "GOLD",
        "join_date": "2022-03-10",
        "is_active": True,
        "email_verified": True,
        "phone_verified": False,
        "has_orders": True,
        "account_age_days": 650,
        "password": "secret123",  # 敏感信息，应该被过滤
        "ssn": "123-45-6789"  # 敏感信息，应该被过滤
    },
    {
        "id": 67890,
        "first_name": "Alice",
        "last_name": "Smith",
        "email": "alice.smith@gmail.com",
        "phone": "5559876543",
        "birth_date": "1985-12-20",
        "street": "456 oak avenue",
        "city": "los angeles",
        "state": "ca",
        "membership_tier": "silver",
        "is_active": False,
        "email_verified": True,
        "has_orders": False
        # 缺失一些字段
    }
]

for i, user_data in enumerate(test_users, 1):
    print(f"\n📥 原始用户数据 {i}:")
    print(f"  姓名: {user_data.get('first_name', '')} {user_data.get('last_name', '')}")
    print(f"  邮箱: {user_data.get('email', '')}")

    result = user_mapper.invoke(user_data)

    print(f"📤 标准化后数据 {i}:")
    print(f"  用户ID: {result['user_id']}")
    print(f"  全名: {result['full_name']}")
    print(f"  邮箱: {result['email']}")
    print(f"  电话: {result['phone']}")
    print(f"  年龄: {result['age']} ({result['age_group']})")
    print(f"  地址: {result['address']}")
    print(f"  会员: {result['membership']}")
    print(f"  行为分: {result['behavior_score']}")
    print("-" * 40)


# 内置Runnable函数：3、RunnableSequence - 序列组合
# RunnableSequence 用于按特定顺序执行依赖任务，每个步骤的输出作为下一个步骤的输入。适用于有严格先后顺序的业务流程。
# 电商订单处理需要严格按照以下顺序执行：
# 1、验证订单信息
# 2、检查库存
# 3、计算价格（含折扣）
# 4、处理支付
# 5、更新库存
# 6、发送确认通知
# 这些步骤必须按顺序执行，因为后续步骤依赖前序步骤的结果。
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time
from datetime import datetime


# 模拟订单处理函数
def validate_order(order_data):
    """1. 验证订单信息"""
    print("📋 步骤1: 验证订单信息...")
    time.sleep(0.2)

    required_fields = ["user_id", "items", "shipping_address"]
    for field in required_fields:
        if field not in order_data:
            raise ValueError(f"缺少必要字段: {field}")

    # 模拟验证逻辑
    if len(order_data["items"]) == 0:
        raise ValueError("订单商品不能为空")

    return {
        **order_data,
        "validation_status": "success",
        "validated_at": datetime.now().isoformat()
    }


def check_inventory(validated_order):
    """2. 检查库存"""
    print("📦 步骤2: 检查库存...")
    time.sleep(0.3)

    inventory_status = {}
    for item in validated_order["items"]:
        # 模拟库存检查
        item_id = item["product_id"]
        inventory_status[item_id] = {
            "available": True,  # 模拟库存充足
            "stock_count": 100,
            "reserved": False
        }

    return {
        **validated_order,
        "inventory_status": inventory_status,
        "all_items_available": all(status["available"] for status in inventory_status.values())
    }


def calculate_pricing(inventory_checked_order):
    """3. 计算价格和折扣"""
    print("💰 步骤3: 计算价格...")
    time.sleep(0.2)

    if not inventory_checked_order["all_items_available"]:
        raise ValueError("部分商品库存不足")

    # 计算商品总价
    subtotal = sum(item["price"] * item["quantity"] for item in inventory_checked_order["items"])

    # 应用折扣逻辑
    user_tier = inventory_checked_order.get("user_tier", "standard")
    discount_rate = 0.1 if user_tier == "vip" else 0.0
    discount_amount = subtotal * discount_rate

    # 计算运费
    shipping_fee = 0 if subtotal > 200 else 15

    total_amount = subtotal - discount_amount + shipping_fee

    return {
        **inventory_checked_order,
        "pricing_details": {
            "subtotal": subtotal,
            "discount_rate": discount_rate,
            "discount_amount": discount_amount,
            "shipping_fee": shipping_fee,
            "total_amount": total_amount
        }
    }


def process_payment(priced_order):
    """4. 处理支付"""
    print("💳 步骤4: 处理支付...")
    time.sleep(0.5)

    payment_success = True  # 模拟支付成功
    payment_id = f"PAY_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if not payment_success:
        raise ValueError("支付处理失败")

    return {
        **priced_order,
        "payment_status": "completed",
        "payment_id": payment_id,
        "paid_at": datetime.now().isoformat()
    }


def update_inventory(paid_order):
    """5. 更新库存"""
    print("🔄 步骤5: 更新库存...")
    time.sleep(0.3)

    # 模拟库存扣减
    for item in paid_order["items"]:
        print(f"   - 扣减商品 {item['product_id']} 库存: {item['quantity']} 件")

    return {
        **paid_order,
        "inventory_updated": True,
        "inventory_update_time": datetime.now().isoformat()
    }


def send_confirmation(processed_order):
    """6. 发送确认通知"""
    print("📧 步骤6: 发送确认通知...")
    time.sleep(0.1)

    # 生成确认信息
    confirmation_message = f"""
🎉 订单处理完成！

订单号: {processed_order.get('order_id', 'N/A')}
用户: {processed_order['user_id']}
总金额: ${processed_order['pricing_details']['total_amount']:.2f}
支付状态: {processed_order['payment_status']}
支付ID: {processed_order['payment_id']}

商品清单:
{chr(10).join(f"  - {item['name']} x{item['quantity']}" for item in processed_order['items'])}

感谢您的购买！
"""

    return {
        **processed_order,
        "confirmation_sent": True,
        "confirmation_message": confirmation_message,
        "completed_at": datetime.now().isoformat()
    }


# 创建顺序处理链
# first必选，有且只接受一个Runnable对象
# middle可选，可多个步骤，多个步骤会按列表顺序依次执行（不支持并行）
# last必选，有且只接受一个Runnable对象
order_processing_sequence = RunnableSequence(
    first=RunnableLambda(validate_order),
    middle=[
        RunnableLambda(check_inventory),
        RunnableLambda(calculate_pricing),
        RunnableLambda(process_payment),
        RunnableLambda(update_inventory)
    ],
    last=RunnableLambda(send_confirmation)
)

# 测试订单处理
print("🚀 开始处理订单...")
start_time = time.time()

sample_order = {
    "order_id": "ORD_20240120001",
    "user_id": "user_12345",
    "user_tier": "vip",
    "items": [
        {"product_id": "P001", "name": "智能手机", "price": 2999, "quantity": 1},
        {"product_id": "P002", "name": "无线耳机", "price": 399, "quantity": 2}
    ],
    "shipping_address": "北京市朝阳区xxx街道"
}

try:
    result = order_processing_sequence.invoke(sample_order)

    end_time = time.time()
    print(f"\n⏱️ 订单处理总时间: {end_time - start_time:.2f}秒")

    print("\n" + "=" * 60)
    print("✅ 订单处理结果:")
    print("=" * 60)
    print(result["confirmation_message"])

    print("\n📊 处理状态汇总:")
    print(f"  - 验证状态: {result.get('validation_status', 'N/A')}")
    print(f"  - 库存状态: {result.get('all_items_available', 'N/A')}")
    print(f"  - 支付状态: {result.get('payment_status', 'N/A')}")
    print(f"  - 库存更新: {result.get('inventory_updated', 'N/A')}")
    print(f"  - 确认发送: {result.get('confirmation_sent', 'N/A')}")

except Exception as e:
    print(f"❌ 订单处理失败: {e}")


# 内置Runnable函数：4、RunnableBranch - 条件分支
# 用于根据条件动态选择执行路径，实现业务流程的分支逻辑。适用于需要根据不同条件执行不同处理的场景。

# # Example：智能客服路由系统:
# 订单问题 → 订单客服
# 支付问题 → 支付客服
# 售后问题 → 售后客服
# 产品咨询 → 产品专家
# 其他问题 → 普通客服
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import re


def create_customer_service_router():
    """创建智能客服路由系统"""

    # 定义条件判断函数
    def is_order_issue(user_input):
        """判断是否为订单问题"""
        order_keywords = ['订单', '发货', '物流', '配送', '取消订单', '修改订单']
        return any(keyword in user_input for keyword in order_keywords)

    def is_payment_issue(user_input):
        """判断是否为支付问题"""
        payment_keywords = ['支付', '付款', '退款', '金额', '扣款', '支付宝', '微信支付']
        return any(keyword in user_input for keyword in payment_keywords)

    def is_after_sales_issue(user_input):
        """判断是否为售后问题"""
        after_sales_keywords = ['退货', '换货', '维修', '保修', '售后', '质量問題']
        return any(keyword in user_input for keyword in after_sales_keywords)

    def is_product_inquiry(user_input):
        """判断是否为产品咨询"""
        product_keywords = ['产品', '商品', '规格', '功能', '使用方法', '参数']
        return any(keyword in user_input for keyword in product_keywords)

    # 定义不同分支的处理逻辑
    def handle_order_issue(user_input):
        """处理订单问题"""
        return {
            "department": "订单客服",
            "priority": "高",
            "assigned_agent": "订单专家",
            "response": "您的订单问题已转接给订单专家，请稍等...",
            "estimated_wait_time": "2分钟",
            "specialty": "物流跟踪、订单修改、取消处理"
        }

    def handle_payment_issue(user_input):
        """处理支付问题"""
        return {
            "department": "支付客服",
            "priority": "紧急",
            "assigned_agent": "支付风控专员",
            "response": "支付问题已转接给支付风控专员，请准备好订单信息...",
            "estimated_wait_time": "1分钟",
            "specialty": "退款处理、支付失败、金额核对"
        }

    def handle_after_sales_issue(user_input):
        """处理售后问题"""
        return {
            "department": "售后客服",
            "priority": "中",
            "assigned_agent": "售后顾问",
            "response": "售后问题已转接，我们将为您处理退货换货事宜...",
            "estimated_wait_time": "3分钟",
            "specialty": "退货流程、换货政策、质量检测"
        }

    def handle_product_inquiry(user_input):
        """处理产品咨询"""
        return {
            "department": "产品专家",
            "priority": "低",
            "assigned_agent": "产品顾问",
            "response": "产品咨询已转接给产品专家，将为您详细介绍产品信息...",
            "estimated_wait_time": "1分钟",
            "specialty": "产品功能、规格参数、使用指导"
        }

    def handle_general_inquiry(user_input):
        """处理一般咨询"""
        return {
            "department": "普通客服",
            "priority": "低",
            "assigned_agent": "客服代表",
            "response": "您的问题已转接给客服代表，我们将尽力为您解答...",
            "estimated_wait_time": "5分钟",
            "specialty": "一般咨询、账户问题、操作指导"
        }

    # 默认结构：RunnableBranch(
    #     (条件1, 处理1),
    #     (条件2, 处理2),
    #     (条件3, 处理3),
    #     ...
    #     默认处理  # 没有条件元组
    # )
    # 执行逻辑：
    # 1、按顺序检查每个条件
    # 2、第一个为 True 的条件，执行对应的处理，跳过所有后续条件检查和处理，直接返回该处理函数的结果
    # 3、如果所有条件都为 False，执行默认处理

    # Tips：注意顺序，更严格的条件 或 检测成本低的条件，放在最前面

    router_branch = RunnableBranch(
        (lambda x: is_order_issue(x["user_input"]), RunnableLambda(lambda x: handle_order_issue(x["user_input"]))),
        (lambda x: is_payment_issue(x["user_input"]), RunnableLambda(lambda x: handle_payment_issue(x["user_input"]))),
        (lambda x: is_after_sales_issue(x["user_input"]), RunnableLambda(lambda x: handle_after_sales_issue(x["user_input"]))),
        (lambda x: is_product_inquiry(x["user_input"]), RunnableLambda(lambda x: handle_product_inquiry(x["user_input"]))),
        RunnableLambda(lambda x: handle_general_inquiry(x["user_input"]))  # 默认分支
    )

    return router_branch


# 使用智能客服路由
print("🤖 智能客服路由系统")
print("=" * 50)

service_router = create_customer_service_router()

# 测试不同用户问题
test_cases = [
    {"user_input": "我的订单什么时候能发货？", "user_id": "U001"},
    {"user_input": "支付失败了但是钱扣了怎么办？", "user_id": "U002"},
    {"user_input": "想退货需要什么流程？", "user_id": "U003"},
    {"user_input": "这个产品的电池能用多久？", "user_id": "U004"},
    {"user_input": "你们公司的地址在哪里？", "user_id": "U005"}
]

for case in test_cases:
    print(f"\n用户问题: {case['user_input']}")
    result = service_router.invoke(case)
    print(f"路由结果: {result['department']}")
    print(f"响应: {result['response']}")
    print(f"预计等待: {result['estimated_wait_time']}")
    print("-" * 40)


# 6、RunnableConfig
# 1、监控逻辑与业务逻辑分离
# 2、所有链使用相同的监控标准

# 业务场景1：生产环境监控和日志记录
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging
import time


class MonitoringCallbackHandler(BaseCallbackHandler):
    """生产环境监控回调"""

    # 链执行开始时触发：链的序列化信息、输入数据、开始时间
    def on_chain_start(self, serialized, inputs, **kwargs):
        logging.info(f"🔗 链开始: {serialized.get('name', 'unknown')}")
        logging.info(f"📥 输入: {inputs}")
        self.start_time = time.time()

    # # 链执行结束时触发：执行时间、输出结果
    def on_chain_end(self, outputs, **kwargs):
        duration = time.time() - self.start_time
        logging.info(f"✅ 链结束: 耗时 {duration:.2f}秒")
        logging.info(f"📤 输出: {outputs}")

    # 链执行错误时触发：记录错误信息
    def on_chain_error(self, error, **kwargs):
        logging.error(f"❌ 链错误: {error}")

    # 监控LLM、记录token使用量
    def on_llm_start(self, serialized, prompts, **kwargs):
        logging.info(f"🤖 LLM调用开始: {len(prompts)}个提示词")

    def on_llm_end(self, response, **kwargs):
        logging.info(f"✅ LLM调用完成")
        if hasattr(response, 'usage'):
            logging.info(f"📊 Token使用: {response.usage}")


# 创建生产环境配置
# callbacks：1、非侵入式监控，不修改业务代码 2、支持多个callbacks
# tags：标签系统：环境标识、业务分类、优先级
    # 不同环境使用不同配置：tag=production，用production_config
    #                   tag=customer-facing,用customer_facing_config,并发数可放宽
# metadata：版本、部署ID、环境、负责团队、联系人等
# run_name:唯一标识实例、区分不同版本的链、支撑不同版本对比和基准测试
production_config = RunnableConfig(
    callbacks=[MonitoringCallbackHandler()],
    tags=["production"],
    metadata={
        "environment": "production",
        "service_version": "2.1.0",
        "deployment_id": "dep-2024-01-20",
        "team": "ai-platform"
    },
    run_name="customer_service_chain_v2",
    max_concurrency=10,  # 限制并发数，保护后端服务
    recursion_limit=50  # 防止递归过深
)

customer_facing_config= RunnableConfig(
    callbacks=[MonitoringCallbackHandler()],
    tags=["customer-facing"],
    metadata={
        "environment": "production",
        "service_version": "2.1.0",
        "deployment_id": "dep-2024-01-20",
        "team": "ai-platform"
    },
    run_name="customer_service_chain_v2",
    max_concurrency=30,  # 放宽并发数
    recursion_limit=50
)

# 使用配置的业务链
customer_service_chain = (
        ChatPromptTemplate.from_template("回答用户问题: {question}") | ChatOpenAI(model="gpt-3.5-turbo")
)

# 执行时传入配置
question = "我的订单状态如何？"

result = customer_service_chain.invoke(
    {"question": question},
    config=production_config
)

# 同一链，不同环境使用不同配置
development_result = customer_service_chain.invoke(
    config=production_config)      # 开发环境

production_result = customer_service_chain.invoke(
    config=customer_facing_config)  # 生产环境


# 7、RunnablePassthrough数据传递器
# 8、RunnableRetry重试机制
# LangGraph
# LangSmith
# 高级模式 - 如代理（Agents）、工具（Tools）、记忆（Memory）、Prompt、
# 部署模式 - 容器化、扩缩容、负载均衡
