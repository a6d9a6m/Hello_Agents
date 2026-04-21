"""测试新的记忆工具接口"""

from src.tools.builtin.memory_tool import MemoryTool, MemoryOperation
from src.memory.base import MemoryType


def test_new_memory_tool():
    """测试新的记忆工具接口"""
    print("=== 测试新的记忆工具接口 ===")
    
    tool = MemoryTool()
    
    # 测试添加记忆
    print("\n1. 测试添加记忆...")
    result = tool.execute(
        action="add",
        content="用户喜欢在早上喝咖啡，每天一杯",
        memory_type="working",
        importance=0.8,
        emotion="positive",
        tags=["coffee", "morning", "habit"],
        location="home",
        source="user_conversation"
    )
    print(f"   结果: {result}")
    
    # 测试添加更多记忆
    print("\n2. 测试添加更多记忆...")
    result = tool.execute(
        action="add",
        content="项目截止日期是下周五，需要完成AI系统开发",
        memory_type="working",
        importance=0.9,
        emotion="urgent",
        tags=["project", "deadline", "AI"],
        location="office",
        source="project_management"
    )
    print(f"   结果: {result}")
    
    result = tool.execute(
        action="add",
        content="昨天在会议室讨论了新的产品需求",
        memory_type="episodic",
        importance=0.7,
        emotion="neutral",
        tags=["meeting", "product", "requirements"],
        location="conference_room",
        source="meeting_notes"
    )
    print(f"   结果: {result}")
    
    # 测试搜索记忆
    print("\n3. 测试搜索记忆...")
    result = tool.execute(
        action="search",
        content="咖啡",
        memory_types=["working", "episodic"],
        search_mode="hybrid",
        limit=5,
        min_importance=0.5
    )
    print(f"   结果: {result}")
    
    # 测试关键词搜索
    print("\n4. 测试关键词搜索...")
    result = tool.execute(
        action="search",
        content="项目",
        search_mode="keyword",
        limit=3
    )
    print(f"   结果: {result}")
    
    # 测试获取摘要
    print("\n5. 测试获取摘要...")
    result = tool.execute(
        action="summary",
        memory_type="working"
    )
    print(f"   结果: {result}")
    
    # 测试获取统计
    print("\n6. 测试获取统计...")
    result = tool.execute(
        action="stats"
    )
    print(f"   结果: {result[:200]}...")  # 截断输出
    
    # 测试遗忘记忆（基于重要性）
    print("\n7. 测试遗忘记忆（基于重要性）...")
    result = tool.execute(
        action="forget",
        forget_strategy="importance_based",
        importance_threshold=0.85  # 删除重要性低于0.85的记忆
    )
    print(f"   结果: {result}")
    
    # 测试整合记忆
    print("\n8. 测试整合记忆...")
    result = tool.execute(
        action="consolidate",
        importance_threshold=0.75  # 将重要性≥0.75的工作记忆转为情景记忆
    )
    print(f"   结果: {result}")
    
    # 测试更新记忆
    print("\n9. 测试更新记忆...")
    # 首先添加一个测试记忆
    add_result = tool.execute(
        action="add",
        content="测试更新功能的记忆",
        memory_type="working",
        importance=0.6
    )
    print(f"   添加结果: {add_result}")
    
    # 提取记忆ID（简化处理）
    if "ID：" in add_result:
        memory_id = add_result.split("ID：")[1].split("\n")[0].strip()
        print(f"   记忆ID: {memory_id}")
        
        # 更新记忆
        result = tool.execute(
            action="update",
            memory_id=memory_id,
            memory_type="working",
            updates={"importance": 0.8, "emotion": "updated"}
        )
        print(f"   更新结果: {result}")
    
    # 测试删除记忆
    print("\n10. 测试删除记忆...")
    result = tool.execute(
        action="remove",
        memory_id=memory_id if 'memory_id' in locals() else "test_id",
        memory_type="working"
    )
    print(f"   结果: {result}")
    
    # 测试清空所有记忆
    print("\n11. 测试清空所有记忆...")
    result = tool.execute(
        action="clear_all"
    )
    print(f"   结果: {result}")
    
    print("\n[完成] 新的记忆工具接口测试完成")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    tool = MemoryTool()
    
    # 测试旧的参数格式（通过execute方法）
    print("\n1. 测试旧的存储操作...")
    result = tool.execute(
        action="add",  # 对应旧的"store"
        content="向后兼容测试内容",
        memory_type="working"
    )
    print(f"   结果: {result}")
    
    print("\n2. 测试旧的检索操作...")
    result = tool.execute(
        action="search",  # 对应旧的"retrieve"
        content="测试",
        memory_type="working",
        limit=5
    )
    print(f"   结果: {result}")
    
    print("\n3. 测试旧的统计操作...")
    result = tool.execute(
        action="stats"  # 对应旧的"stats"
    )
    print(f"   结果: {result[:100]}...")
    
    print("\n[完成] 向后兼容性测试完成")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    tool = MemoryTool()
    
    # 测试空内容
    print("\n1. 测试空内容添加...")
    result = tool.execute(
        action="add",
        content="",
        memory_type="working"
    )
    print(f"   结果: {result}")
    
    # 测试无效操作
    print("\n2. 测试无效操作...")
    result = tool.execute(
        action="invalid_action",
        content="测试"
    )
    print(f"   结果: {result}")
    
    # 测试不存在的记忆ID
    print("\n3. 测试不存在的记忆ID...")
    result = tool.execute(
        action="remove",
        memory_id="non_existent_id",
        memory_type="working"
    )
    print(f"   结果: {result}")
    
    # 测试多种遗忘策略
    print("\n4. 测试多种遗忘策略...")
    strategies = ["importance_based", "time_based", "capacity_based", "combined"]
    for strategy in strategies:
        result = tool.execute(
            action="forget",
            forget_strategy=strategy
        )
        print(f"   策略 '{strategy}': {result}")
    
    print("\n[完成] 边界情况测试完成")


if __name__ == "__main__":
    print("开始测试新的记忆工具接口...")
    print("=" * 50)
    
    try:
        test_new_memory_tool()
        test_backward_compatibility()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("[完成] 所有测试完成！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()