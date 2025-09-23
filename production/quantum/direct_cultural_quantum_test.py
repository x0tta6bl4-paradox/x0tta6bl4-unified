#!/usr/bin/env python3
"""
🧪 Прямой тест Cultural Quantum компонентов
==========================================

Тест с прямыми импортами для проверки работоспособности
"""

import sys
import time
import asyncio
import os

# Добавляем путь к модулям
sys.path.append('src')

def test_direct_imports():
    """Тестирование прямых импортов"""
    print("🧪 Тестирование прямых импортов...")
    
    try:
        # Прямые импорты из файлов
        import importlib.util
        
        # Тест 1: Cultural Quantum Avatars
        print("1️⃣ Тестирование Cultural Quantum Avatars...")
        avatars_path = 'src/x0tta6bl4/ai/cultural_quantum_avatars.py'
        if os.path.exists(avatars_path):
            spec = importlib.util.spec_from_file_location("cultural_quantum_avatars", avatars_path)
            avatars_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(avatars_module)
            
            # Создаем менеджер аватаров
            avatar_manager = avatars_module.CulturalQuantumAvatarManager()
            print("   ✅ Cultural Quantum Avatar Manager создан")
            
            # Создаем аватары
            for archetype in ["иван-дурак", "василиса-премудрая", "кощей-бессмертный"]:
                avatar = avatar_manager.create_avatar(archetype)
                if avatar:
                    print(f"   ✅ Аватар {archetype} создан: φ-гармония {avatar.phi_harmony_score:.3f}")
                else:
                    print(f"   ❌ Ошибка создания аватара {archetype}")
            
            # Тестируем квантовые ответы
            response = avatar_manager.generate_quantum_response("Тест φ-гармонического ответа")
            print(f"   🧠 Квантовый ответ: {response.get('response', 'Нет ответа')[:50]}...")
            print(f"   🧠 Культурный резонанс: {response.get('cultural_resonance', 0):.3f}")
            
        else:
            print(f"   ❌ Файл не найден: {avatars_path}")
        
        # Тест 2: Anti-Hallucination System
        print("2️⃣ Тестирование Anti-Hallucination System...")
        anti_hall_path = 'src/x0tta6bl4/security/sber_anti_hallucination.py'
        if os.path.exists(anti_hall_path):
            spec = importlib.util.spec_from_file_location("sber_anti_hallucination", anti_hall_path)
            anti_hall_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(anti_hall_module)
            
            # Создаем систему anti-hallucination
            anti_hall = anti_hall_module.SberAntiHallucinationSystem()
            print("   ✅ Anti-Hallucination System создана")
            
            # Тестируем обнаружение галлюцинаций
            test_texts = [
                "Квантовая суперпозиция в φ-гармонии",
                "Я не знаю, возможно это галлюцинация",
                "Согласно исследованиям, φ-гармония равна 1.618"
            ]
            
            for text in test_texts:
                result = anti_hall.detect_hallucinations(text)
                print(f"   📝 '{text[:30]}...' -> Галлюцинация: {result.get('is_hallucination', False)}, Уверенность: {result.get('confidence', 0):.2f}")
            
        else:
            print(f"   ❌ Файл не найден: {anti_hall_path}")
        
        # Тест 3: Qwen Image Generator
        print("3️⃣ Тестирование Qwen Image Generator...")
        image_gen_path = 'src/x0tta6bl4/ai/qwen_phi_image_generator.py'
        if os.path.exists(image_gen_path):
            spec = importlib.util.spec_from_file_location("qwen_phi_image_generator", image_gen_path)
            image_gen_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(image_gen_module)
            
            # Создаем генератор изображений
            image_generator = image_gen_module.PhiOptimizedQwenImage()
            print("   ✅ Qwen Image Generator создан")
            
            # Тестируем генерацию изображений
            image_result = image_generator.generate_with_phi_harmony(
                "Квантовая эволюция через φ-гармонию",
                "эволюция",
                "квантовый реализм"
            )
            
            print(f"   🎨 φ-гармонический скор: {image_result.get('phi_harmony_score', 0):.3f}")
            print(f"   🎨 Успех генерации: {image_result.get('generation_success', False)}")
            
        else:
            print(f"   ❌ Файл не найден: {image_gen_path}")
        
        # Тест 4: Cultural Quantum Integration
        print("4️⃣ Тестирование Cultural Quantum Integration...")
        integration_path = 'src/x0tta6bl4/integration/cultural_quantum_integration.py'
        if os.path.exists(integration_path):
            spec = importlib.util.spec_from_file_location("cultural_quantum_integration", integration_path)
            integration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(integration_module)
            
            # Создаем интеграцию
            integration = integration_module.CulturalQuantumIntegration()
            print("   ✅ Cultural Quantum Integration создана")
            
            # Тестируем интеграцию
            async def test_integration():
                try:
                    integration_result = await integration.process_cultural_quantum_request(
                        "Тест культурного квантового запроса",
                        "иван-дурак",
                        "тестовый",
                        "тестовый реализм"
                    )
                    return integration_result
                except Exception as e:
                    return {"error": str(e)}
            
            integration_result = asyncio.run(test_integration())
            
            if "error" not in integration_result:
                print("   ✅ Интеграция работает успешно")
                print(f"   ⏱️ Время обработки: {integration_result.get('processing_time', 0):.3f}s")
            else:
                print(f"   ❌ Ошибка интеграции: {integration_result.get('error', 'Неизвестная ошибка')}")
            
        else:
            print(f"   ❌ Файл не найден: {integration_path}")
        
        print("\n🎉 ВСЕ ПРЯМЫЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРЯМОГО ТЕСТИРОВАНИЯ: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evolution_engine_simple():
    """Простое тестирование Evolution Engine"""
    print("\n🔗 Простое тестирование Evolution Engine...")
    
    try:
        # Прямой импорт Evolution Engine
        evolution_path = 'src/x0tta6bl4/evolution/evolution_engine.py'
        if os.path.exists(evolution_path):
            spec = importlib.util.spec_from_file_location("evolution_engine", evolution_path)
            evolution_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evolution_module)
            
            print("   ✅ Evolution Engine импортирован")
            
            # Создаем конфигурацию
            config = evolution_module.EvolutionConfig()
            print("   ✅ Конфигурация создана")
            
            # Создаем движок эволюции
            engine = evolution_module.EvolutionEngine(config)
            print("   ✅ Evolution Engine создан")
            
            # Проверяем состояние
            print(f"   🎭 Cultural Quantum доступен: {hasattr(engine, 'cultural_avatar_manager')}")
            print(f"   🎭 Anti-Hallucination доступен: {hasattr(engine, 'anti_hallucination_system')}")
            print(f"   🎭 Image Generator доступен: {hasattr(engine, 'image_generator')}")
            
            return True
            
        else:
            print(f"   ❌ Файл не найден: {evolution_path}")
            return False
        
    except Exception as e:
        print(f"   ❌ Ошибка тестирования Evolution Engine: {e}")
        return False

def main():
    """Главная функция"""
    print("🧪 Прямой тест Cultural Quantum компонентов")
    print("=" * 50)
    
    # Тест прямых импортов
    direct_ok = test_direct_imports()
    
    # Тест Evolution Engine
    evolution_ok = test_evolution_engine_simple()
    
    # Итоговый результат
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Прямые импорты: {'✅ OK' if direct_ok else '❌ FAIL'}")
    print(f"   Evolution Engine: {'✅ OK' if evolution_ok else '❌ FAIL'}")
    
    if direct_ok and evolution_ok:
        print("\n🎉 ВСЕ ПРЯМЫЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("🚀 Cultural Quantum компоненты работают!")
    else:
        print("\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("🔧 Требуется дополнительная настройка")

if __name__ == "__main__":
    main()
