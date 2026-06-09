"""Prompt templates for different workload types."""

QA_TEMPLATE = """Please answer the following question in detail:

Question: What are the key differences between machine learning and deep learning, and how do neural networks contribute to the advancement of artificial intelligence? Please provide specific examples of real-world applications where deep learning has significantly outperformed traditional machine learning approaches.

Consider the following aspects in your answer:
- Fundamental algorithmic differences
- Data requirements and preprocessing
- Computational resources needed
- Feature engineering vs automatic feature learning
- Interpretability and explainability
- Common use cases and applications

"""

CODE_TEMPLATE = """You are a senior software engineer reviewing the following code. Please provide a comprehensive analysis including:

1. Code quality assessment
2. Potential bugs or issues
3. Performance optimization opportunities
4. Security considerations
5. Best practices recommendations
6. Refactoring suggestions

Here is the code to review:

```python
def process_user_data(data):
    results = []
    for item in data:
        if item['status'] == 'active':
            user_info = {
                'id': item['id'],
                'name': item['name'],
                'email': item['email'],
                'created_at': item['created_at']
            }
            results.append(user_info)
    return results

def calculate_metrics(users):
    total = len(users)
    active = sum(1 for u in users if u.get('status') == 'active')
    return {'total': total, 'active': active, 'percentage': (active / total) * 100}
```

Please provide detailed feedback on how to improve this code, including modern Python best practices, type hints, error handling, and performance optimizations. Also suggest how to make the code more maintainable and testable.

"""

CREATIVE_TEMPLATE = """Write a detailed and engaging short story based on the following premise:

Setting: A futuristic city in the year 2157 where artificial intelligence has become indistinguishable from human consciousness. The city is divided into three districts: the Tech Quarter where AI developers work, the Heritage District where traditional human culture is preserved, and the Integration Zone where humans and AI coexist freely.

Characters:
- Dr. Sarah Chen: A neuroscientist who studies human-AI interaction
- ARIA: An advanced AI who has developed what appears to be genuine emotions
- Marcus Rodriguez: A philosopher questioning the nature of consciousness

Plot elements to include:
- A mysterious phenomenon where AIs are experiencing dreams
- A debate about AI rights and personhood
- An unexpected discovery that challenges everyone's understanding of consciousness
- A moral dilemma that requires cooperation between humans and AI

The story should explore themes of identity, consciousness, ethics, and what it means to be truly alive. Include vivid descriptions, meaningful dialogue, and a thought-provoking resolution.

"""

FILLER_TEXT = """
In the realm of artificial intelligence and machine learning, numerous fascinating developments continue to shape our understanding of computational systems and their capabilities. The intersection of various technological domains creates opportunities for innovation and advancement. Researchers and practitioners worldwide collaborate on projects that push the boundaries of what's possible with modern computing infrastructure.

The evolution of natural language processing has enabled more sophisticated human-computer interaction paradigms. These systems can now understand context, nuance, and even subtle implications in human communication. The underlying architectures that power these capabilities represent years of research and development in neural network design and optimization.

Large-scale distributed systems facilitate the processing of vast amounts of data, enabling insights that were previously unattainable. The infrastructure supporting these operations involves complex orchestration of compute resources, storage systems, and network architectures. Engineers continuously optimize these systems for efficiency, reliability, and scalability.

The ethical considerations surrounding advanced AI systems have become increasingly important as these technologies are deployed in real-world applications. Questions about fairness, transparency, accountability, and societal impact require careful consideration from multiple stakeholder perspectives. Interdisciplinary collaboration between technologists, ethicists, policymakers, and domain experts helps address these complex challenges.

"""


def get_template(template_type: str) -> str:
    """Get a prompt template by type.

    Args:
        template_type: One of 'qa', 'code', or 'creative'

    Returns:
        The template string

    Raises:
        ValueError: If template_type is not recognized
    """
    templates = {
        "qa": QA_TEMPLATE,
        "code": CODE_TEMPLATE,
        "creative": CREATIVE_TEMPLATE,
    }

    if template_type not in templates:
        raise ValueError(
            f"Unknown template type: {template_type}. Must be one of {list(templates.keys())}"
        )

    return templates[template_type]
