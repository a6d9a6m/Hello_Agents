"""演示新的记忆工具功能"""

from src.tools.builtin.memory_tool import MemoryTool


def demo_memory_tool():
    """演示记忆工具的主要功能"""
    print("=" * 60)
    print("新的记忆工具功能演示")
    print("=" * 60)
    
    tool = MemoryTool()
    
    print("\n1. 添加丰富的记忆上下文")
    print("-" * 40)
    
    # 添加工作记忆
    memories = [
        {
            "content": "用户Alice喜欢在星巴克喝拿铁咖啡",
            "memory_type": "working",
            "importance": 0.8,
            "emotion": "positive",
            "tags": ["coffee", "starbucks", "preference"],
            "location": "starbucks_downtown",
            "source": "user_profile"
        },
        {
            "content": "项目Alpha需要在3月15日前完成原型开发",
            "memory_type": "working", 
            "importance": 0.9,
            "emotion": "urgent",
            "tags": ["project", "deadline", "prototype"],
            "location": "office",
            "source": "project_plan"
        },
        {
            "content": "昨天团队会议讨论了新的UI设计规范",
            "memory_type": "episodic",
            "importance": 0.7,
            "emotion": "neutral",
            "tags": ["meeting", "design", "team"],
            "location": "conference_room_a",
            "source": "meeting_minutes"
        }
    ]
    
    for i, memory in enumerate(memories, 1):
        result = tool.execute(action="add", **memory)
        print(f"记忆{i}: {result.split('...')[0]}")
    
    print("\n2. 智能搜索功能")
    print("-" * 40)
    
    # 混合搜索
    print("搜索'咖啡'（混合模式）:")
    result = tool.execute(
        action="search",
        content="咖啡",
        search_mode="hybrid",
        limit=3
    )
    print(result)
    
    # 关键词搜索
    print("\n搜索'项目'（关键词模式）:")
    result = tool.execute(
        action="search",
        content="项目",
        search_mode="keyword",
        limit=2
    )
    print(result)
    
    print("\n3. 记忆摘要和统计")
    print("-" * 40)
    
    # 工作记忆摘要
    print("工作记忆摘要:")
    result = tool.execute(action="summary", memory_type="working")
    print(result)
    
    # 完整统计
    print("\n完整统计信息:")
    result = tool.execute(action="stats")
    # 只显示部分统计信息
    lines = result.split('\n')
    for line in lines[:15]:
        print(line)
    
    print("\n4. 记忆管理功能")
    print("-" * 40)
    
    # 基于重要性的遗忘
    print("遗忘重要性<0.75的记忆:")
    result = tool.execute(
        action="forget",
        forget_strategy="importance_based",
        importance_threshold=0.75
    )
    print(result)
    
    # 记忆整合
    print("\n整合工作记忆（重要性≥0.7转为情景记忆）:")
    result = tool.execute(
        action="consolidate",
        importance_threshold=0.7
    )
    print(result)
    
    print("\n5. 更新后的统计")
    print("-" * 40)
    result = tool.execute(action="stats")
    lines = result.split('\n')
    for line in lines[:10]:
        print(line)
    
    print("\n6. 清空所有记忆")
    print("-" * 40)
    result = tool.execute(action="clear_all")
    print(result)
    
    print("\n" + "=" * 60)
    print("演示完成！新的记忆工具支持：")
    print("√ 丰富的上下文信息（重要性、情感、标签等）")
    print("√ 多种搜索模式（关键词、语义、混合）")
    print("√ 智能遗忘策略（基于重要性、时间、容量）")
    print("√ 记忆整合（工作记忆→情景记忆）")
    print("√ 详细的统计和摘要")
    print("√ 向后兼容旧接口")
    print("=" * 60)


if __name__ == "__main__":
    demo_memory_tool()