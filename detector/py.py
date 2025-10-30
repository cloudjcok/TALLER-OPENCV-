#!/usr/bin/env python3
"""
Script para verificar que los archivos Haar Cascade estén disponibles
"""
import cv2
import os

def verificar_cascades():
    print("=" * 60)
    print("VERIFICACIÓN DE HAAR CASCADE FILES")
    print("=" * 60)
    
    # 1. Verificar la ruta base
    cascade_path = cv2.data.haarcascades
    print(f"\n📁 Ruta de Haar Cascades:")
    print(f"   {cascade_path}")
    
    # 2. Verificar que la carpeta existe
    if os.path.exists(cascade_path):
        print("   ✅ La carpeta existe")
    else:
        print("   ❌ La carpeta NO existe")
        return
    
    # 3. Archivos necesarios para el proyecto
    required_files = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml',
        'haarcascade_smile.xml'
    ]
    
    print("\n🔍 Archivos requeridos para el proyecto:")
    all_exist = True
    for filename in required_files:
        full_path = os.path.join(cascade_path, filename)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {filename}")
        if exists:
            size = os.path.getsize(full_path)
            print(f"      Tamaño: {size:,} bytes")
        else:
            all_exist = False
    
    # 4. Listar todos los archivos disponibles
    print("\n📦 Todos los archivos .xml disponibles:")
    xml_files = [f for f in os.listdir(cascade_path) if f.endswith('.xml')]
    for filename in sorted(xml_files):
        print(f"   • {filename}")
    
    # 5. Probar carga de clasificadores
    print("\n🧪 Probando carga de clasificadores:")
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("   ✅ Face Cascade cargado correctamente")
        print(f"      Vacío: {face_cascade.empty()}")
    except Exception as e:
        print(f"   ❌ Error cargando Face Cascade: {e}")
    
    try:
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("   ✅ Eye Cascade cargado correctamente")
        print(f"      Vacío: {eye_cascade.empty()}")
    except Exception as e:
        print(f"   ❌ Error cargando Eye Cascade: {e}")
    
    try:
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        print("   ✅ Smile Cascade cargado correctamente")
        print(f"      Vacío: {smile_cascade.empty()}")
    except Exception as e:
        print(f"   ❌ Error cargando Smile Cascade: {e}")
    
    # 6. Verificar versión de OpenCV
    print(f"\n🔧 Versión de OpenCV: {cv2.__version__}")
    
    # 7. Resultado final
    print("\n" + "=" * 60)
    if all_exist:
        print("✅ TODOS LOS ARCHIVOS NECESARIOS ESTÁN DISPONIBLES")
        print("   Tu proyecto debería funcionar correctamente")
    else:
        print("⚠️  FALTAN ALGUNOS ARCHIVOS")
        print("   Reinstala OpenCV: pip install --upgrade opencv-python")
    print("=" * 60)

if __name__ == "__main__":
    try:
        verificar_cascades()
    except Exception as e:
        print(f"\n❌ Error durante la verificación: {e}")
        print("\nIntenta reinstalar OpenCV:")
        print("   pip uninstall opencv-python")
        print("   pip install opencv-python")