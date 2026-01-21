#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
SCRIPT DE MIGRATION AUTOMATIQUE DICOM → PACS ORTHANC
============================================================================
Projet : Segmentation Multi-Organes Pulmonaire
Auteur : ENSAM - Intelligence Artificielle
Date   : Janvier 2026

Description :
    Ce script parcourt récursivement un dossier local, identifie tous les
    fichiers DICOM et les envoie automatiquement vers un serveur PACS Orthanc
    via son API REST.

Fonctionnalités :
    - Détection automatique des fichiers DICOM (avec ou sans extension .dcm)
    - Envoi parallèle pour de meilleures performances
    - Barre de progression en temps réel
    - Gestion robuste des erreurs de connexion
    - Logs détaillés avec statistiques finales
    - Mode dry-run pour tester sans envoyer

Usage :
    python migration_pacs.py --source ./DATA --url http://localhost:8042
    python migration_pacs.py --source ./DATA --dry-run
    python migration_pacs.py --help

============================================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports externes
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("❌ Erreur: Le module 'requests' n'est pas installé.")
    print("   Installation: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  Module 'tqdm' non installé. Barre de progression simplifiée.")
    print("   Installation optionnelle: pip install tqdm")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PACSConfig:
    """Configuration du serveur PACS Orthanc"""
    url: str = "http://localhost:8042"
    username: str = "admin"
    password: str = "orthanc123"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_workers: int = 4

# ============================================================================
# CLASSES PRINCIPALES
# ============================================================================

class DicomScanner:
    """Scanner de fichiers DICOM dans un répertoire"""
    
    # Signature magique DICOM (octets 128-132)
    DICOM_MAGIC = b'DICM'
    DICOM_MAGIC_OFFSET = 128
    
    # Extensions DICOM connues
    DICOM_EXTENSIONS = {'.dcm', '.dicom', '.dic', '.ima', '.img'}
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)
    
    def is_dicom_file(self, filepath: Path) -> bool:
        """
        Vérifie si un fichier est un fichier DICOM valide.
        Détection par signature magique (plus fiable que l'extension).
        """
        try:
            # Vérifier la taille minimale
            if filepath.stat().st_size < 132:
                return False
            
            # Lire la signature DICOM
            with open(filepath, 'rb') as f:
                f.seek(self.DICOM_MAGIC_OFFSET)
                magic = f.read(4)
            
            return magic == self.DICOM_MAGIC
            
        except (IOError, OSError) as e:
            self.logger.debug(f"Impossible de lire {filepath}: {e}")
            return False
    
    def scan(self) -> List[Path]:
        """
        Parcourt récursivement le répertoire et retourne la liste
        des fichiers DICOM trouvés.
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Répertoire introuvable: {self.source_dir}")
        
        if not self.source_dir.is_dir():
            raise NotADirectoryError(f"Ce n'est pas un répertoire: {self.source_dir}")
        
        dicom_files = []
        
        self.logger.info(f"🔍 Scan du répertoire: {self.source_dir}")
        
        # Parcours récursif
        for filepath in self.source_dir.rglob('*'):
            if filepath.is_file():
                # Vérifier d'abord par extension (rapide)
                if filepath.suffix.lower() in self.DICOM_EXTENSIONS:
                    if self.is_dicom_file(filepath):
                        dicom_files.append(filepath)
                # Sinon vérifier par signature (fichiers sans extension)
                elif filepath.suffix == '' or filepath.suffix.lower() not in {'.txt', '.json', '.xml', '.csv', '.py', '.md'}:
                    if self.is_dicom_file(filepath):
                        dicom_files.append(filepath)
        
        self.logger.info(f"📁 {len(dicom_files)} fichiers DICOM trouvés")
        return dicom_files


class OrthancClient:
    """Client pour l'API REST Orthanc"""
    
    def __init__(self, config: PACSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = self._create_session()
        
        # Statistiques
        self.stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'total_bytes': 0
        }
    
    def _create_session(self) -> requests.Session:
        """Crée une session HTTP avec retry automatique"""
        session = requests.Session()
        
        # Configuration des retries
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.parallel_workers,
            pool_maxsize=self.config.parallel_workers * 2
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Authentification
        session.auth = (self.config.username, self.config.password)
        
        return session
    
    def test_connection(self) -> bool:
        """Teste la connexion au serveur Orthanc"""
        try:
            url = f"{self.config.url}/system"
            response = self.session.get(url, timeout=self.config.timeout)
            
            if response.status_code == 200:
                info = response.json()
                self.logger.info(f"✅ Connexion réussie à Orthanc")
                self.logger.info(f"   Version: {info.get('Version', 'N/A')}")
                self.logger.info(f"   Nom: {info.get('Name', 'N/A')}")
                return True
            else:
                self.logger.error(f"❌ Erreur HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.logger.error(f"❌ Impossible de se connecter à {self.config.url}")
            self.logger.error("   Vérifiez que le serveur Orthanc est démarré.")
            return False
        except requests.exceptions.Timeout:
            self.logger.error(f"❌ Timeout lors de la connexion")
            return False
        except Exception as e:
            self.logger.error(f"❌ Erreur inattendue: {e}")
            return False
    
    def upload_dicom(self, filepath: Path) -> Tuple[bool, str]:
        """
        Envoie un fichier DICOM vers Orthanc.
        
        Returns:
            Tuple (success: bool, message: str)
        """
        try:
            # Lire le fichier
            with open(filepath, 'rb') as f:
                dicom_data = f.read()
            
            # Envoyer via l'API REST
            url = f"{self.config.url}/instances"
            headers = {'Content-Type': 'application/dicom'}
            
            response = self.session.post(
                url,
                data=dicom_data,
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                instance_id = result.get('ID', 'N/A')
                status = result.get('Status', 'Success')
                
                self.stats['success'] += 1
                self.stats['total_bytes'] += len(dicom_data)
                
                if status == 'AlreadyStored':
                    return True, f"Déjà stocké (ID: {instance_id[:8]}...)"
                return True, f"Succès (ID: {instance_id[:8]}...)"
            
            elif response.status_code == 400:
                self.stats['failed'] += 1
                return False, "Fichier DICOM invalide"
            
            elif response.status_code == 401:
                self.stats['failed'] += 1
                return False, "Authentification échouée"
            
            else:
                self.stats['failed'] += 1
                return False, f"Erreur HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            self.stats['failed'] += 1
            return False, "Erreur de connexion"
        
        except requests.exceptions.Timeout:
            self.stats['failed'] += 1
            return False, "Timeout"
        
        except IOError as e:
            self.stats['failed'] += 1
            return False, f"Erreur lecture fichier: {e}"
        
        except Exception as e:
            self.stats['failed'] += 1
            return False, f"Erreur inattendue: {e}"
    
    def get_statistics(self) -> dict:
        """Retourne les statistiques du serveur Orthanc"""
        try:
            url = f"{self.config.url}/statistics"
            response = self.session.get(url, timeout=self.config.timeout)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}


class DicomMigrator:
    """Classe principale pour la migration DICOM vers PACS"""
    
    def __init__(self, config: PACSConfig, source_dir: str, dry_run: bool = False):
        self.config = config
        self.source_dir = source_dir
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        
        self.scanner = DicomScanner(source_dir)
        self.client = OrthancClient(config)
    
    def run(self) -> bool:
        """
        Exécute la migration complète.
        
        Returns:
            True si succès, False sinon
        """
        start_time = time.time()
        
        # Bannière
        self._print_banner()
        
        # Mode dry-run
        if self.dry_run:
            self.logger.warning("🔸 MODE DRY-RUN: Aucun fichier ne sera envoyé")
        
        # Test de connexion
        if not self.dry_run:
            self.logger.info("📡 Test de connexion au serveur PACS...")
            if not self.client.test_connection():
                return False
        
        # Scan des fichiers
        try:
            dicom_files = self.scanner.scan()
        except (FileNotFoundError, NotADirectoryError) as e:
            self.logger.error(f"❌ {e}")
            return False
        
        if not dicom_files:
            self.logger.warning("⚠️  Aucun fichier DICOM trouvé!")
            return True
        
        # Migration
        self.logger.info(f"📤 Début de la migration de {len(dicom_files)} fichiers...")
        
        if self.dry_run:
            self._dry_run_migration(dicom_files)
        else:
            self._execute_migration(dicom_files)
        
        # Statistiques finales
        elapsed = time.time() - start_time
        self._print_summary(elapsed)
        
        return self.client.stats['failed'] == 0
    
    def _print_banner(self):
        """Affiche la bannière du script"""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║           MIGRATION DICOM → PACS ORTHANC                         ║
║           Projet Segmentation Multi-Organes                      ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def _execute_migration(self, dicom_files: List[Path]):
        """Exécute la migration avec barre de progression"""
        
        if TQDM_AVAILABLE:
            # Avec tqdm (belle barre de progression)
            with tqdm(total=len(dicom_files), desc="Migration", unit="fichier") as pbar:
                for filepath in dicom_files:
                    success, message = self.client.upload_dicom(filepath)
                    
                    # Mise à jour de la description
                    status = "✓" if success else "✗"
                    pbar.set_postfix_str(f"{status} {filepath.name[:30]}")
                    pbar.update(1)
        else:
            # Sans tqdm (logs simples)
            total = len(dicom_files)
            for i, filepath in enumerate(dicom_files, 1):
                success, message = self.client.upload_dicom(filepath)
                
                status = "✓" if success else "✗"
                percent = (i / total) * 100
                
                # Afficher toutes les 10 fichiers ou en cas d'erreur
                if i % 10 == 0 or not success:
                    print(f"[{percent:5.1f}%] {status} {filepath.name} - {message}")
    
    def _dry_run_migration(self, dicom_files: List[Path]):
        """Mode dry-run: liste les fichiers sans les envoyer"""
        print("\n📋 Fichiers DICOM qui seraient migrés:\n")
        
        total_size = 0
        for i, filepath in enumerate(dicom_files[:50], 1):  # Limiter à 50 pour l'affichage
            size = filepath.stat().st_size
            total_size += size
            print(f"  {i:4d}. {filepath.relative_to(self.source_dir)} ({size/1024:.1f} KB)")
        
        if len(dicom_files) > 50:
            print(f"  ... et {len(dicom_files) - 50} autres fichiers")
            # Calculer la taille totale
            for filepath in dicom_files[50:]:
                total_size += filepath.stat().st_size
        
        print(f"\n📊 Total: {len(dicom_files)} fichiers ({total_size/1024/1024:.1f} MB)")
    
    def _print_summary(self, elapsed: float):
        """Affiche le résumé de la migration"""
        stats = self.client.stats
        
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA MIGRATION")
        print("="*60)
        
        if not self.dry_run:
            print(f"  ✅ Succès:     {stats['success']:,}")
            print(f"  ❌ Échecs:     {stats['failed']:,}")
            print(f"  📦 Données:    {stats['total_bytes']/1024/1024:.2f} MB")
            print(f"  ⏱️  Durée:      {elapsed:.1f} secondes")
            
            if stats['success'] > 0 and elapsed > 0:
                rate = stats['success'] / elapsed
                print(f"  🚀 Débit:      {rate:.1f} fichiers/sec")
            
            # Statistiques serveur
            server_stats = self.client.get_statistics()
            if server_stats:
                print("\n📈 Statistiques Serveur PACS:")
                print(f"  • Patients:  {server_stats.get('CountPatients', 'N/A')}")
                print(f"  • Études:    {server_stats.get('CountStudies', 'N/A')}")
                print(f"  • Séries:    {server_stats.get('CountSeries', 'N/A')}")
                print(f"  • Instances: {server_stats.get('CountInstances', 'N/A')}")
        
        print("="*60)
        
        if stats['failed'] == 0:
            print("🎉 Migration terminée avec succès!")
        else:
            print(f"⚠️  Migration terminée avec {stats['failed']} erreur(s)")


# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure le système de logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    
    # Logger principal
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def parse_arguments():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Migration automatique de fichiers DICOM vers PACS Orthanc',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --source ./DATA
  %(prog)s --source ./DATA --url http://192.168.1.100:8042
  %(prog)s --source ./DATA --dry-run
  %(prog)s --source ./DATA --user admin --password secret
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='./DATA',
        help='Répertoire source contenant les fichiers DICOM (défaut: ./DATA)'
    )
    
    parser.add_argument(
        '--url', '-u',
        type=str,
        default='http://localhost:8042',
        help='URL du serveur Orthanc (défaut: http://localhost:8042)'
    )
    
    parser.add_argument(
        '--user',
        type=str,
        default='admin',
        help='Nom d\'utilisateur Orthanc (défaut: admin)'
    )
    
    parser.add_argument(
        '--password', '-p',
        type=str,
        default='orthanc123',
        help='Mot de passe Orthanc (défaut: orthanc123)'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=30,
        help='Timeout en secondes (défaut: 30)'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Mode simulation: liste les fichiers sans les envoyer'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbeux avec logs détaillés'
    )
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_arguments()
    
    # Configuration du logging
    setup_logging(args.verbose)
    
    # Configuration PACS
    config = PACSConfig(
        url=args.url,
        username=args.user,
        password=args.password,
        timeout=args.timeout
    )
    
    # Exécution de la migration
    migrator = DicomMigrator(
        config=config,
        source_dir=args.source,
        dry_run=args.dry_run
    )
    
    success = migrator.run()
    
    # Code de retour
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
