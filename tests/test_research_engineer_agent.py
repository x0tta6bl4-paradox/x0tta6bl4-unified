"""
Тесты для Research Engineer Agent
"""

import pytest
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock
from research.research_engineer_agent import ResearchEngineerAgent, ResearchResult, StatisticalAnalysis, Documentation


class TestResearchEngineerAgent:
    """Тесты для Research Engineer Agent"""

    @pytest.fixture
    def agent(self):
        """Фикстура для создания агента"""
        return ResearchEngineerAgent()

    @pytest.fixture
    def sample_research_data(self):
        """Фикстура с тестовыми данными исследования"""
        return {
            "experiment_id": "test_exp_001",
            "algorithm": "shor",
            "problem_size": 15,
            "quantum_time": 0.5,
            "classical_time": 10.0,
            "speedup_factor": 20.0,
            "accuracy": 0.95,
            "success_rate": 0.9,
            "provider": "ibm",
            "metadata": {"test": True}
        }

    @pytest.fixture
    def sample_doc_data(self):
        """Фикстура с тестовыми данными документации"""
        return {
            "title": "Test Technical Specification",
            "description": "Test quantum algorithm specification",
            "algorithm": "shor",
            "problem_size": 15,
            "accuracy": 0.95
        }

    def test_agent_initialization(self, agent):
        """Тест инициализации агента"""
        assert agent.name == "research_engineer_agent"
        assert agent.status == "initialized"
        assert isinstance(agent.research_database, list)
        assert isinstance(agent.documentation_database, list)
        assert isinstance(agent.publication_pipeline, list)
        assert hasattr(agent, 'logger')

    @pytest.mark.asyncio
    async def test_agent_initialize(self, agent):
        """Тест метода initialize"""
        result = await agent.initialize()
        assert result == True
        assert agent.status == "operational"

        # Проверка создания директорий
        assert os.path.exists("research/data")
        assert os.path.exists("research/experiments")
        assert os.path.exists("research/papers")
        assert os.path.exists("research/presentations")

    @pytest.mark.asyncio
    async def test_agent_health_check(self, agent):
        """Тест метода health_check"""
        await agent.initialize()
        result = await agent.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_agent_get_status(self, agent):
        """Тест метода get_status"""
        await agent.initialize()
        status = await agent.get_status()

        assert status["name"] == "research_engineer_agent"
        assert "status" in status
        assert "research_results_count" in status
        assert "documentation_count" in status
        assert "publication_pipeline_length" in status
        assert "coordination_api" in status

    @pytest.mark.asyncio
    async def test_agent_shutdown(self, agent):
        """Тест метода shutdown"""
        await agent.initialize()
        result = await agent.shutdown()
        assert result == True
        assert agent.status == "shutdown"

    @pytest.mark.asyncio
    async def test_analyze_research_results(self, agent, sample_research_data):
        """Тест анализа результатов исследований"""
        await agent.initialize()

        result = await agent.analyze_research_results(sample_research_data)

        assert "experiment_analyzed" in result
        assert "statistical_analysis" in result
        assert "trend_analysis" in result
        assert "classical_comparison" in result
        assert "quantum_advantage" in result
        assert "performance_benchmark" in result
        assert "recommendations" in result

        # Проверка добавления в базу данных
        assert len(agent.research_database) == 1
        assert agent.research_database[0].experiment_id == "test_exp_001"

    @pytest.mark.asyncio
    async def test_generate_documentation_technical(self, agent, sample_doc_data):
        """Тест генерации технической документации"""
        await agent.initialize()

        result = await agent.generate_documentation("technical", sample_doc_data)

        assert "documentation" in result
        assert "file_path" in result
        assert "word_count" in result
        assert result["documentation"]["doc_type"] == "technical"
        assert result["documentation"]["format"] == "markdown"

        # Проверка добавления в базу данных
        assert len(agent.documentation_database) == 1

    @pytest.mark.asyncio
    async def test_generate_documentation_scientific(self, agent, sample_doc_data):
        """Тест генерации научной документации"""
        await agent.initialize()

        result = await agent.generate_documentation("scientific", sample_doc_data)

        assert result["documentation"]["doc_type"] == "scientific"
        assert result["documentation"]["format"] == "latex"

    @pytest.mark.asyncio
    async def test_generate_documentation_presentation(self, agent, sample_doc_data):
        """Тест генерации презентационной документации"""
        await agent.initialize()

        result = await agent.generate_documentation("presentation", sample_doc_data)

        assert result["documentation"]["doc_type"] == "presentation"
        assert result["documentation"]["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_create_scientific_publications(self, agent, sample_research_data):
        """Тест создания научных публикаций"""
        await agent.initialize()

        result = await agent.create_scientific_publications("Quantum Advantage Research", sample_research_data)

        assert "publication" in result
        assert "supplementary_materials" in result
        assert "estimated_review_time" in result
        assert "target_journals" in result

        # Проверка добавления в pipeline
        assert len(agent.publication_pipeline) == 1
        assert agent.publication_pipeline[0]["topic"] == "Quantum Advantage Research"

    @pytest.mark.asyncio
    async def test_generate_presentations(self, agent, sample_doc_data):
        """Тест генерации презентаций"""
        await agent.initialize()

        result = await agent.generate_presentations("conference", sample_doc_data)

        assert "presentation" in result
        assert "structure" in result
        assert "visualizations" in result
        assert "file_path" in result
        assert "presentation_tips" in result

        # Проверка структуры презентации
        structure = result["structure"]
        assert "title_slide" in structure
        assert "introduction" in structure
        assert "methodology" in structure
        assert "results" in structure
        assert "conclusion" in structure

    def test_statistical_analysis_dataclass(self):
        """Тест StatisticalAnalysis dataclass"""
        analysis = StatisticalAnalysis(
            mean_speedup=15.5,
            std_speedup=2.3,
            confidence_interval=(12.1, 18.9),
            p_value=0.001,
            effect_size=1.2,
            sample_size=20,
            normality_test={"statistic": 0.95, "p_value": 0.8}
        )

        assert analysis.mean_speedup == 15.5
        assert analysis.p_value == 0.001
        assert analysis.sample_size == 20

    def test_research_result_dataclass(self, sample_research_data):
        """Тест ResearchResult dataclass"""
        result = ResearchResult(**sample_research_data, timestamp=1234567890.0)

        assert result.experiment_id == "test_exp_001"
        assert result.algorithm == "shor"
        assert result.speedup_factor == 20.0
        assert result.accuracy == 0.95

    def test_documentation_dataclass(self):
        """Тест Documentation dataclass"""
        doc = Documentation(
            doc_type="technical",
            title="Test Doc",
            content="# Test Content",
            format="markdown",
            metadata={"test": True},
            created_at=1234567890.0
        )

        assert doc.doc_type == "technical"
        assert doc.title == "Test Doc"
        assert doc.format == "markdown"

    @pytest.mark.asyncio
    async def test_register_collaborator(self, agent):
        """Тест регистрации сотрудника"""
        mock_agent = MagicMock()
        result = await agent.register_collaborator("ai_engineer", mock_agent)

        assert result == True
        assert agent.coordination_api["ai_engineer"] == mock_agent

    @pytest.mark.asyncio
    async def test_request_ai_analysis_no_agent(self, agent):
        """Тест запроса AI анализа без агента"""
        result = await agent.request_ai_analysis({"test": "data"})

        assert "error" in result
        assert result["error"] == "AI Engineer недоступен"

    @pytest.mark.asyncio
    async def test_request_quantum_expertise_no_agent(self, agent):
        """Тест запроса quantum экспертизы без агента"""
        result = await agent.request_quantum_expertise("test_topic")

        assert "error" in result
        assert result["error"] == "Quantum Engineer недоступен"

    @pytest.mark.asyncio
    async def test_coordinate_research_project(self, agent):
        """Тест координации исследовательского проекта"""
        project_config = {
            "name": "Test Project",
            "objectives": ["Test objective"],
            "topic": "quantum computing"
        }

        result = await agent.coordinate_research_project(project_config)

        assert "project_plan" in result
        assert "ai_support" in result
        assert "quantum_support" in result
        assert result["project_plan"]["name"] == "Test Project"

    @pytest.mark.asyncio
    async def test_multiple_experiments_analysis(self, agent):
        """Тест анализа множественных экспериментов"""
        await agent.initialize()

        # Добавление нескольких экспериментов
        experiments = [
            {
                "experiment_id": f"exp_{i}",
                "algorithm": "shor",
                "problem_size": 10 + i,
                "quantum_time": 0.1 * i,
                "classical_time": 1.0 * i,
                "speedup_factor": 10.0 + i,
                "accuracy": 0.9 + 0.01 * i,
                "success_rate": 0.85 + 0.01 * i,
                "provider": "ibm",
                "metadata": {}
            }
            for i in range(1, 6)  # 5 экспериментов
        ]

        for exp in experiments:
            await agent.analyze_research_results(exp)

        # Проверка что все эксперименты добавлены
        assert len(agent.research_database) == 5

        # Выполнение анализа
        analysis_result = await agent.analyze_research_results(experiments[0])  # Повторный анализ

        statistical = analysis_result["statistical_analysis"]
        assert statistical["sample_size"] == 6  # 5 + 1 новый

    def test_target_journal_selection(self, agent):
        """Тест выбора целевого журнала"""
        # Высокое ускорение - Nature
        result = agent._select_target_journal({"speedup": 150})
        assert result == "Nature"

        # Среднее ускорение - Science
        result = agent._select_target_journal({"speedup": 50})
        assert result == "Science"

        # Низкое ускорение - Quantum Information Processing
        result = agent._select_target_journal({"speedup": 1.5})
        assert result == "Quantum Information Processing"

    def test_presentation_tips(self, agent):
        """Тест советов по презентации"""
        tips = agent._get_presentation_tips("conference")
        assert isinstance(tips, list)
        assert len(tips) > 0

        tips = agent._get_presentation_tips("internal")
        assert isinstance(tips, list)

        tips = agent._get_presentation_tips("unknown")
        assert isinstance(tips, list)

    @pytest.mark.asyncio
    async def test_database_persistence(self, agent, tmp_path):
        """Тест сохранения и загрузки базы данных"""
        await agent.initialize()

        # Добавление тестовых данных
        await agent.analyze_research_results({
            "experiment_id": "persist_test",
            "algorithm": "test",
            "problem_size": 10,
            "quantum_time": 0.5,
            "classical_time": 5.0,
            "speedup_factor": 10.0,
            "accuracy": 0.9,
            "success_rate": 0.8,
            "provider": "test",
            "metadata": {"persistent": True}
        })

        # Сохранение
        await agent._save_research_database()

        # Проверка файла
        assert os.path.exists("research/data/research_database.json")

        # Создание нового агента и загрузка
        new_agent = ResearchEngineerAgent()
        await new_agent._load_research_database()

        # Проверка загрузки данных
        assert len(new_agent.research_database) == 1
        assert new_agent.research_database[0].experiment_id == "persist_test"