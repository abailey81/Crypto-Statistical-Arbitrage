"""
Whale Alert Large Transaction Tracker

validated collector for tracking large cryptocurrency transactions
across multiple blockchains. Real-time whale movement monitoring and
exchange flow analysis.

===============================================================================
WHALE ALERT OVERVIEW
===============================================================================

Whale Alert is the leading service for tracking large cryptocurrency
transactions in real-time. Used by traders, researchers, and media
for market intelligence and breaking transaction news.

Key Features:
    - Real-time large transaction alerts
    - Exchange deposit/withdrawal tracking
    - Known wallet labels (exchanges, funds, etc.)
    - Multi-chain support (BTC, ETH, XRP, etc.)
    - Historical transaction data

===============================================================================
API SPECIFICATIONS
===============================================================================

Base URL: https://api.whale-alert.io/v1

Authentication:
    - API Key as query parameter

Rate Limits by Tier:
    ============ ============== ================ ===============
    Tier Requests/min Monthly Limit Min TX Value
    ============ ============== ================ ===============
    Free 10 1,000 $500K
    Basic 30 10,000 $100K
    Professional 60 100,000 $10K
    Enterprise Custom Unlimited Custom
    ============ ============== ================ ===============

Minimum Transaction Value:
    - Free tier: $500,000 USD minimum
    - Paid tiers: Lower thresholds available

===============================================================================
DATA FIELDS
===============================================================================

Transaction Fields:
    - hash: Transaction hash
    - blockchain: Source blockchain
    - symbol: Cryptocurrency symbol
    - timestamp: Transaction time
    - amount: Transaction amount
    - amount_usd: USD value at time
    - transaction_type: transfer, mint, burn, lock, unlock

Address Fields:
    - from/to address
    - from/to owner (entity name if known)
    - from/to owner_type (exchange, unknown, etc.)

===============================================================================
STATISTICAL ARBITRAGE APPLICATIONS
===============================================================================

Market Signals:
    - Large exchange inflows -> Potential selling pressure
    - Large exchange outflows -> Accumulation signal
    - Unknown -> Exchange -> Potential dump
    - Exchange -> Unknown -> OTC or accumulation

Risk Assessment:
    - Whale concentration monitoring
    - Exchange flow imbalance detection
    - Large transfer pattern analysis

Timing Signals:
    - Volume-weighted whale activity
    - Exchange-specific flow tracking
    - Cross-exchange arbitrage detection

Version: 2.0.0
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp
import pandas as pd
import numpy as np

# =============================================================================
# ENUMS
# =============================================================================

class Blockchain(Enum):
    """Supported blockchains."""
    BITCOIN = 'bitcoin'
    ETHEREUM = 'ethereum'
    RIPPLE = 'ripple'
    TRON = 'tron'
    BINANCE_CHAIN = 'binancechain'
    STELLAR = 'stellar'
    NEO = 'neo'
    EOS = 'eos'
    LITECOIN = 'litecoin'
    BITCOIN_CASH = 'bitcoincash'
    DOGECOIN = 'dogecoin'

class TransactionType(Enum):
    """Transaction types."""
    TRANSFER = 'transfer'
    MINT = 'mint'
    BURN = 'burn'
    LOCK = 'lock'
    UNLOCK = 'unlock'

class EntityType(Enum):
    """Known entity types."""
    EXCHANGE = 'exchange'
    UNKNOWN = 'unknown'
    OTC = 'otc'
    CUSTODIAN = 'custodian'
    FUND = 'fund'
    PRIVATE = 'private'
    WHALE = 'whale'

class FlowType(Enum):
    """Exchange flow classification."""
    EXCHANGE_INFLOW = 'exchange_inflow'
    EXCHANGE_OUTFLOW = 'exchange_outflow'
    EXCHANGE_TO_EXCHANGE = 'exchange_to_exchange'
    WALLET_TO_WALLET = 'wallet_to_wallet'
    UNKNOWN = 'unknown'

class WhaleSize(Enum):
    """Whale size classification."""
    MEGA_WHALE = 'mega_whale' # > $100M
    LARGE_WHALE = 'large_whale' # $10M - $100M
    WHALE = 'whale' # $1M - $10M
    DOLPHIN = 'dolphin' # $500K - $1M
    FISH = 'fish' # < $500K

class MarketImpact(Enum):
    """Estimated market impact."""
    EXTREME = 'extreme' # Could move market significantly
    HIGH = 'high' # Notable market impact expected
    MODERATE = 'moderate' # Some market impact possible
    LOW = 'low' # Minimal market impact

class SentimentSignal(Enum):
    """Sentiment signal from transaction."""
    VERY_BULLISH = 'very_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    VERY_BEARISH = 'very_bearish'

class AlertPriority(Enum):
    """Alert priority level."""
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class WhaleTransaction:
    """Large cryptocurrency transaction with analytics."""
    timestamp: datetime
    blockchain: str
    symbol: str
    tx_hash: str
    amount: float
    amount_usd: float
    transaction_type: str
    from_address: str
    from_owner: Optional[str] = None
    from_owner_type: Optional[str] = None
    to_address: str = ''
    to_owner: Optional[str] = None
    to_owner_type: Optional[str] = None
    
    @property
    def flow_type(self) -> FlowType:
        """Classify transaction flow type."""
        from_is_exchange = self.from_owner_type == 'exchange'
        to_is_exchange = self.to_owner_type == 'exchange'
        
        if from_is_exchange and not to_is_exchange:
            return FlowType.EXCHANGE_OUTFLOW
        elif not from_is_exchange and to_is_exchange:
            return FlowType.EXCHANGE_INFLOW
        elif from_is_exchange and to_is_exchange:
            return FlowType.EXCHANGE_TO_EXCHANGE
        elif self.from_owner_type and self.to_owner_type:
            return FlowType.WALLET_TO_WALLET
        return FlowType.UNKNOWN
    
    @property
    def whale_size(self) -> WhaleSize:
        """Classify whale size."""
        if self.amount_usd >= 100_000_000:
            return WhaleSize.MEGA_WHALE
        elif self.amount_usd >= 10_000_000:
            return WhaleSize.LARGE_WHALE
        elif self.amount_usd >= 1_000_000:
            return WhaleSize.WHALE
        elif self.amount_usd >= 500_000:
            return WhaleSize.DOLPHIN
        return WhaleSize.FISH
    
    @property
    def market_impact(self) -> MarketImpact:
        """Estimate market impact."""
        if self.amount_usd >= 100_000_000:
            return MarketImpact.EXTREME
        elif self.amount_usd >= 50_000_000:
            return MarketImpact.HIGH
        elif self.amount_usd >= 10_000_000:
            return MarketImpact.MODERATE
        return MarketImpact.LOW
    
    @property
    def sentiment_signal(self) -> SentimentSignal:
        """Derive sentiment signal from transaction."""
        flow = self.flow_type
        size = self.whale_size
        
        if flow == FlowType.EXCHANGE_INFLOW:
            if size in [WhaleSize.MEGA_WHALE, WhaleSize.LARGE_WHALE]:
                return SentimentSignal.VERY_BEARISH
            return SentimentSignal.BEARISH
        elif flow == FlowType.EXCHANGE_OUTFLOW:
            if size in [WhaleSize.MEGA_WHALE, WhaleSize.LARGE_WHALE]:
                return SentimentSignal.VERY_BULLISH
            return SentimentSignal.BULLISH
        return SentimentSignal.NEUTRAL
    
    @property
    def alert_priority(self) -> AlertPriority:
        """Determine alert priority."""
        if self.amount_usd >= 100_000_000:
            return AlertPriority.CRITICAL
        elif self.amount_usd >= 50_000_000:
            return AlertPriority.HIGH
        elif self.amount_usd >= 10_000_000:
            return AlertPriority.MEDIUM
        return AlertPriority.LOW
    
    @property
    def is_exchange_inflow(self) -> bool:
        """Check if exchange deposit."""
        return self.flow_type == FlowType.EXCHANGE_INFLOW
    
    @property
    def is_exchange_outflow(self) -> bool:
        """Check if exchange withdrawal."""
        return self.flow_type == FlowType.EXCHANGE_OUTFLOW
    
    @property
    def is_significant(self) -> bool:
        """Check if transaction is significant."""
        return self.amount_usd >= 10_000_000
    
    @property
    def involves_exchange(self) -> bool:
        """Check if involves exchange."""
        return self.from_owner_type == 'exchange' or self.to_owner_type == 'exchange'
    
    @property
    def source_entity(self) -> str:
        """Get source entity name."""
        return self.from_owner or self.from_owner_type or 'unknown'
    
    @property
    def destination_entity(self) -> str:
        """Get destination entity name."""
        return self.to_owner or self.to_owner_type or 'unknown'
    
    @property
    def transfer_description(self) -> str:
        """Get human-readable transfer description."""
        return f"{self.source_entity} -> {self.destination_entity}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'blockchain': self.blockchain, 'symbol': self.symbol, 'tx_hash': self.tx_hash,
            'amount': self.amount, 'amount_usd': self.amount_usd,
            'transaction_type': self.transaction_type,
            'from_owner': self.from_owner, 'from_owner_type': self.from_owner_type,
            'to_owner': self.to_owner, 'to_owner_type': self.to_owner_type,
            'flow_type': self.flow_type.value, 'whale_size': self.whale_size.value,
            'market_impact': self.market_impact.value, 'sentiment_signal': self.sentiment_signal.value,
            'alert_priority': self.alert_priority.value, 'is_exchange_inflow': self.is_exchange_inflow,
            'is_exchange_outflow': self.is_exchange_outflow, 'transfer_description': self.transfer_description,
        }

@dataclass
class ExchangeFlowSummary:
    """Exchange flow summary for a period."""
    timestamp: datetime
    blockchain: str
    exchange: Optional[str] = None
    period_hours: int = 24
    inflow_count: int = 0
    inflow_amount: float = 0.0
    inflow_usd: float = 0.0
    outflow_count: int = 0
    outflow_amount: float = 0.0
    outflow_usd: float = 0.0
    
    @property
    def net_flow(self) -> float:
        """Net flow amount."""
        return self.inflow_amount - self.outflow_amount
    
    @property
    def net_flow_usd(self) -> float:
        """Net flow in USD."""
        return self.inflow_usd - self.outflow_usd
    
    @property
    def total_flow_usd(self) -> float:
        """Total flow volume in USD."""
        return self.inflow_usd + self.outflow_usd
    
    @property
    def flow_direction(self) -> FlowType:
        """Net flow direction."""
        if self.net_flow_usd > 1_000_000:
            return FlowType.EXCHANGE_INFLOW
        elif self.net_flow_usd < -1_000_000:
            return FlowType.EXCHANGE_OUTFLOW
        return FlowType.UNKNOWN
    
    @property
    def sentiment(self) -> SentimentSignal:
        """Overall sentiment from flows."""
        if self.net_flow_usd > 50_000_000:
            return SentimentSignal.VERY_BEARISH
        elif self.net_flow_usd > 10_000_000:
            return SentimentSignal.BEARISH
        elif self.net_flow_usd < -50_000_000:
            return SentimentSignal.VERY_BULLISH
        elif self.net_flow_usd < -10_000_000:
            return SentimentSignal.BULLISH
        return SentimentSignal.NEUTRAL
    
    @property
    def is_accumulation(self) -> bool:
        """Check if accumulation phase."""
        return self.net_flow_usd < -1_000_000
    
    @property
    def is_distribution(self) -> bool:
        """Check if distribution phase."""
        return self.net_flow_usd > 1_000_000
    
    @property
    def activity_level(self) -> str:
        """Classify activity level."""
        total_count = self.inflow_count + self.outflow_count
        if total_count > 100:
            return 'very_high'
        elif total_count > 50:
            return 'high'
        elif total_count > 20:
            return 'moderate'
        elif total_count > 5:
            return 'low'
        return 'minimal'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'blockchain': self.blockchain, 'exchange': self.exchange, 'period_hours': self.period_hours,
            'inflow_usd': self.inflow_usd, 'outflow_usd': self.outflow_usd,
            'net_flow_usd': self.net_flow_usd, 'total_flow_usd': self.total_flow_usd,
            'sentiment': self.sentiment.value, 'is_accumulation': self.is_accumulation,
            'activity_level': self.activity_level,
        }

@dataclass
class TopTransaction:
    """Top whale transaction summary."""
    rank: int
    tx_hash: str
    blockchain: str
    symbol: str
    amount: float
    amount_usd: float
    from_entity: str
    to_entity: str
    flow_type: str
    timestamp: datetime
    
    @property
    def is_top_10(self) -> bool:
        """Check if in top 10."""
        return self.rank <= 10
    
    @property
    def is_mega_transaction(self) -> bool:
        """Check if mega transaction."""
        return self.amount_usd >= 100_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rank': self.rank, 'tx_hash': self.tx_hash, 'blockchain': self.blockchain,
            'symbol': self.symbol, 'amount': self.amount, 'amount_usd': self.amount_usd,
            'from_entity': self.from_entity, 'to_entity': self.to_entity,
            'flow_type': self.flow_type, 'timestamp': self.timestamp.isoformat(),
            'is_mega_transaction': self.is_mega_transaction,
        }

@dataclass
class APIStatus:
    """Whale Alert API status."""
    result: str
    blockchain_count: int = 0
    transaction_count_24h: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if API is healthy."""
        return self.result == 'success'
    
    @property
    def has_rate_limit(self) -> bool:
        """Check if rate limited."""
        return self.rate_limit_remaining is not None and self.rate_limit_remaining <= 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'result': self.result, 'is_healthy': self.is_healthy,
            'blockchain_count': self.blockchain_count,
            'transaction_count_24h': self.transaction_count_24h,
            'rate_limit_remaining': self.rate_limit_remaining,
        }

# =============================================================================
# COLLECTOR CLASS
# =============================================================================

class WhaleAlertCollector:
    """
    Whale Alert large transaction collector.
    
    Features:
    - Large transaction alerts
    - Exchange inflow/outflow tracking
    - Known wallet labels
    - Multi-chain support
    - Historical transaction data
    """
    
    VENUE = 'whale_alert'
    VENUE_TYPE = 'onchain'
    BASE_URL = 'https://api.whale-alert.io/v1'
    
    SUPPORTED_CHAINS = [b.value for b in Blockchain]
    TRANSACTION_TYPES = [t.value for t in TransactionType]
    ENTITY_TYPES = [e.value for e in EntityType]
    
    def __init__(self, config: Dict):
        """Initialize Whale Alert collector."""
        self.api_key = config.get('whale_alert_api_key', config.get('api_key', ''))
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_stats = {'requests': 0, 'records': 0, 'errors': 0}
    
    async def __aenter__(self) -> 'WhaleAlertCollector':
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def fetch_transactions(
        self, start: Optional[int] = None, end: Optional[int] = None,
        min_value: int = 500000, cursor: Optional[str] = None,
        blockchain: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch whale transactions."""
        session = await self._get_session()
        
        params = {
            'api_key': self.api_key,
            'min_value': min_value,
            'limit': 100
        }
        
        if start:
            params['start'] = start
        else:
            params['start'] = int((datetime.utcnow() - timedelta(hours=24)).timestamp())
        
        if end:
            params['end'] = end
        if cursor:
            params['cursor'] = cursor
        
        self.collection_stats['requests'] += 1
        
        try:
            async with session.get(f'{self.BASE_URL}/transactions', params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if data.get('result') != 'success':
                        self.logger.warning(f"API returned: {data.get('message')}")
                        return pd.DataFrame()
                    
                    transactions = data.get('transactions', [])
                    
                    if not transactions:
                        return pd.DataFrame()
                    
                    records = []
                    for tx in transactions:
                        if blockchain and tx.get('blockchain') != blockchain:
                            continue
                        
                        whale_tx = WhaleTransaction(
                            timestamp=pd.to_datetime(tx.get('timestamp'), unit='s', utc=True),
                            blockchain=tx.get('blockchain'),
                            symbol=tx.get('symbol'),
                            tx_hash=tx.get('hash'),
                            amount=float(tx.get('amount', 0)),
                            amount_usd=float(tx.get('amount_usd', 0)),
                            transaction_type=tx.get('transaction_type', 'transfer'),
                            from_address=tx.get('from', {}).get('address', ''),
                            from_owner=tx.get('from', {}).get('owner'),
                            from_owner_type=tx.get('from', {}).get('owner_type'),
                            to_address=tx.get('to', {}).get('address', ''),
                            to_owner=tx.get('to', {}).get('owner'),
                            to_owner_type=tx.get('to', {}).get('owner_type'),
                        )
                        records.append(whale_tx.to_dict())
                    
                    df = pd.DataFrame(records)
                    
                    # Store cursor for pagination
                    if data.get('cursor'):
                        df.attrs['next_cursor'] = data.get('cursor')
                    
                    df['venue'] = self.VENUE
                    self.collection_stats['records'] += len(df)
                    return df
                    
                elif resp.status == 429:
                    self.logger.warning("Rate limit exceeded")
                    self.collection_stats['errors'] += 1
                else:
                    self.collection_stats['errors'] += 1
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching transactions: {e}")
            self.collection_stats['errors'] += 1
            return pd.DataFrame()
    
    async def fetch_historical_transactions(
        self, blockchain: str = 'bitcoin', start_date: str = '2022-01-01',
        end_date: Optional[str] = None, min_value: int = 1000000
    ) -> pd.DataFrame:
        """Fetch historical whale transactions with pagination."""
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(
            end_date or datetime.utcnow().strftime('%Y-%m-%d'), '%Y-%m-%d'
        ).timestamp())
        
        all_transactions = []
        cursor = None
        page = 0
        max_pages = 100
        
        while page < max_pages:
            df = await self.fetch_transactions(
                start=start_ts, end=end_ts, min_value=min_value,
                cursor=cursor, blockchain=blockchain
            )
            
            if df.empty:
                break
            
            all_transactions.append(df)
            cursor = df.attrs.get('next_cursor')
            
            if not cursor:
                break
            
            page += 1
            await asyncio.sleep(3)
        
        if not all_transactions:
            return pd.DataFrame()
        
        result = pd.concat(all_transactions, ignore_index=True)
        result = result.drop_duplicates(subset=['tx_hash'])
        return result
    
    async def fetch_exchange_flows(
        self, blockchain: str = 'bitcoin', hours: int = 24, min_value: int = 500000
    ) -> pd.DataFrame:
        """Fetch exchange inflow/outflow summary."""
        start_ts = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
        
        df = await self.fetch_transactions(start=start_ts, min_value=min_value, blockchain=blockchain)
        
        if df.empty:
            return pd.DataFrame()
        
        # Aggregate by flow type
        summary_data = []
        for flow_type in ['exchange_inflow', 'exchange_outflow']:
            subset = df[df['flow_type'] == flow_type]
            if not subset.empty:
                summary_data.append({
                    'flow_type': flow_type,
                    'amount': subset['amount'].sum(),
                    'amount_usd': subset['amount_usd'].sum(),
                    'tx_count': len(subset),
                    'blockchain': blockchain,
                    'period_hours': hours,
                    'timestamp': pd.Timestamp.utcnow(),
                })
        
        if not summary_data:
            return pd.DataFrame()
        
        summary = pd.DataFrame(summary_data)
        summary['venue'] = self.VENUE
        return summary
    
    async def fetch_exchange_specific_flows(
        self, exchange_name: str, blockchain: str = 'bitcoin', hours: int = 24
    ) -> pd.DataFrame:
        """Fetch flows for a specific exchange."""
        df = await self.fetch_transactions(
            start=int((datetime.utcnow() - timedelta(hours=hours)).timestamp()),
            blockchain=blockchain
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter for specific exchange
        mask = (
            (df['from_owner'].str.lower() == exchange_name.lower()) |
            (df['to_owner'].str.lower() == exchange_name.lower())
        )
        exchange_df = df[mask]
        
        if exchange_df.empty:
            return pd.DataFrame()
        
        inflow = exchange_df[exchange_df['to_owner'].str.lower() == exchange_name.lower()]['amount_usd'].sum()
        outflow = exchange_df[exchange_df['from_owner'].str.lower() == exchange_name.lower()]['amount_usd'].sum()
        
        summary = ExchangeFlowSummary(
            timestamp=datetime.utcnow(), blockchain=blockchain, exchange=exchange_name,
            period_hours=hours, inflow_usd=inflow, outflow_usd=outflow,
            inflow_count=len(exchange_df[exchange_df['is_exchange_inflow'] == True]),
            outflow_count=len(exchange_df[exchange_df['is_exchange_outflow'] == True]),
        )
        
        return pd.DataFrame([summary.to_dict()])
    
    async def fetch_top_transactions(
        self, blockchain: str = 'bitcoin', hours: int = 24, top_n: int = 10
    ) -> pd.DataFrame:
        """Fetch top N largest transactions."""
        df = await self.fetch_transactions(
            start=int((datetime.utcnow() - timedelta(hours=hours)).timestamp()),
            blockchain=blockchain
        )
        
        if df.empty:
            return pd.DataFrame()
        
        top_df = df.nlargest(top_n, 'amount_usd').copy()
        top_df['rank'] = range(1, len(top_df) + 1)
        top_df['is_top_10'] = top_df['rank'] <= 10
        top_df['is_mega_transaction'] = top_df['amount_usd'] >= 100_000_000
        
        return top_df
    
    async def fetch_status(self) -> APIStatus:
        """Fetch API status and rate limit info."""
        session = await self._get_session()
        
        try:
            async with session.get(f'{self.BASE_URL}/status', params={'api_key': self.api_key}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return APIStatus(
                        result=data.get('result', 'error'),
                        blockchain_count=len(data.get('blockchains', [])),
                        transaction_count_24h=data.get('transaction_count_24h', 0),
                    )
                return APIStatus(result='error')
        except Exception as e:
            return APIStatus(result=str(e))
    
    async def fetch_funding_rates(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Whale Alert doesn't provide funding rates."""
        return pd.DataFrame()
    
    async def fetch_ohlcv(self, symbols: List[str], timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Whale Alert doesn't provide OHLCV data."""
        return pd.DataFrame()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {**self.collection_stats, 'venue': self.VENUE}
    
    @staticmethod
    def get_supported_chains() -> List[str]:
        """Get list of supported chains."""
        return [b.value for b in Blockchain]
    
    @staticmethod
    def get_transaction_types() -> List[str]:
        """Get list of transaction types."""
        return [t.value for t in TransactionType]
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

async def test_whale_alert():
    """Test Whale Alert collector."""
    config = {'whale_alert_api_key': ''}
    collector = WhaleAlertCollector(config)
    try:
        print(f"Supported chains: {collector.get_supported_chains()}")
        print(f"Transaction types: {collector.get_transaction_types()}")
        
        # Test dataclasses
        tx = WhaleTransaction(
            timestamp=datetime.utcnow(), blockchain='bitcoin', symbol='BTC',
            tx_hash='abc123', amount=100, amount_usd=5_000_000,
            transaction_type='transfer', from_address='0x1',
            from_owner_type='unknown', to_owner='binance', to_owner_type='exchange'
        )
        print(f"Flow type: {tx.flow_type.value}, whale size: {tx.whale_size.value}")
        print(f"Sentiment: {tx.sentiment_signal.value}, priority: {tx.alert_priority.value}")
        
        status = await collector.fetch_status()
        print(f"API status: {status.is_healthy}")
    finally:
        await collector.close()

if __name__ == '__main__':
    asyncio.run(test_whale_alert())