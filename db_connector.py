"""
Database Connector - Supabase Handler
Centralized database operations with type hints.
MASTER VERSION - Supports all project phases including:
- Granular Working Capital, Fleet, and History (Original)
- Sprint 2: Customer/Site/Machine hierarchy
- Sprint 3: Import system support
- Sprint 4: Ore-type aware wear profiles, Forecasting
- Sprint 5: Pipeline/Prospect management
- Sprint 6: Forecast snapshots & Monte Carlo
- Sprint 7: Expense Assumptions (NEW)
"""
import streamlit as st
import pandas as pd
from supabase import create_client, Client
from typing import List, Dict, Optional, Any
from datetime import datetime


class SupabaseHandler:
    """
    Handler class for Supabase database operations.
    Initializes client from Streamlit secrets and provides typed methods.
    """
    
    def __init__(self):
        """Initialize Supabase client from Streamlit secrets"""
        try:
            url = st.secrets["supabase"]["url"]
            # Support multiple key names
            if "service_role_key" in st.secrets["supabase"]:
                key = st.secrets["supabase"]["service_role_key"]
            elif "anon_key" in st.secrets["supabase"]:
                key = st.secrets["supabase"]["anon_key"]
            else:
                key = st.secrets["supabase"]["key"]
                
            self.client: Client = create_client(url, key)
        except KeyError as e:
            st.error(f"Missing Supabase configuration in secrets: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error connecting to Supabase: {e}")
            st.stop()
    
    # =========================================================================
    # 1. SCENARIO MANAGEMENT
    # =========================================================================
    def get_user_scenarios(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            response = (
                self.client.table("scenarios")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading scenarios: {e}")
            return []
    
    def create_scenario(self, user_id: str, name: str, status: str = "draft") -> Optional[Dict[str, Any]]:
        try:
            response = self.client.table("scenarios").insert({
                "name": name,
                "user_id": user_id,
                "status": status
            }).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            st.error(f"Error creating scenario: {e}")
            return None

    def update_scenario(self, scenario_id: str, user_id: str, name: Optional[str] = None, status: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            update_data = {}
            if name is not None: update_data["name"] = name
            if status is not None: update_data["status"] = status
            if not update_data: return None
            
            response = (
                self.client.table("scenarios")
                .update(update_data)
                .eq("id", scenario_id)
                .eq("user_id", user_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            st.error(f"Error updating scenario: {e}")
            return None
    
    def delete_scenario(self, scenario_id: str, user_id: str) -> bool:
        try:
            self.client.table("scenarios").delete().eq("id", scenario_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting scenario: {e}")
            return False

    # =========================================================================
    # 2. ASSUMPTIONS (JSONB)
    # =========================================================================
    def get_scenario_assumptions(self, scenario_id: str, user_id: str = None) -> Dict[str, Any]:
        try:
            response = self.client.table("assumptions").select("data").eq("scenario_id", scenario_id).execute()
            if response.data:
                return response.data[0].get("data", {})
            return {}
        except Exception as e:
            st.error(f"Error loading assumptions: {e}")
            return {}

    def update_assumptions(self, scenario_id: str, user_id: str, assumptions: Dict[str, Any]) -> bool:
        try:
            existing = self.client.table("assumptions").select("id").eq("scenario_id", scenario_id).execute()
            payload = {"scenario_id": scenario_id, "user_id": user_id, "data": assumptions}
            if existing.data:
                assump_id = existing.data[0]['id']
                self.client.table("assumptions").update({"data": assumptions}).eq("id", assump_id).execute()
            else:
                self.client.table("assumptions").insert(payload).execute()
            return True
        except Exception as e:
            st.error(f"Error saving assumptions: {e}")
            return False

    # =========================================================================
    # 3. HISTORICAL FINANCIALS (Revenue & OpEx)
    # =========================================================================
    def get_historic_financials(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Get monthly P&L data from historic_financials table.
        
        This is the PRIMARY method - queries the table where users import 
        their monthly P&L data (revenue, cogs, gross_profit, opex, ebit).
        """
        try:
            response = (
                self.client.table("historic_financials")  # CORRECT TABLE for monthly P&L
                .select("*")
                .eq("scenario_id", scenario_id)
                .order("month")
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading historic financials: {e}")
            return []

    def get_historical_financials(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Get categorized annual data from historical_financials table (legacy).
        
        This queries a different table with schema: year, category, line_item, amount.
        Use get_historic_financials() for monthly P&L data.
        """
        try:
            response = (
                self.client.table("historical_financials")  # Legacy annual/categorized table
                .select("*")
                .eq("scenario_id", scenario_id)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            # Silently return empty - this table may not exist for all scenarios
            return []

    def upsert_historic_financials(self, scenario_id: str, records: List[Dict[str, Any]]) -> bool:
        """Upsert monthly P&L data to historic_financials table."""
        try:
            if not records: return True
            for r in records:
                r['scenario_id'] = scenario_id
            self.client.table("historic_financials").upsert(records).execute()  # CORRECT TABLE
            return True
        except Exception as e:
            st.error(f"Error saving historic financials: {e}")
            return False

    def upsert_historical_financials(self, scenario_id: str, records: List[Dict[str, Any]]) -> bool:
        """Upsert categorized annual data to historical_financials table (legacy)."""
        try:
            if not records: return True
            for r in records:
                r['scenario_id'] = scenario_id
            self.client.table("historical_financials").upsert(records).execute()
            return True
        except Exception as e:
            st.error(f"Error saving historical financials: {e}")
            return False

    def upsert_historical_financials_by_user(self, user_id: str, records: List[Dict[str, Any]], scenario_id: Optional[str] = None) -> bool:
        try:
            if not records: return True
            for r in records:
                r['user_id'] = user_id
                if scenario_id:
                    r['scenario_id'] = scenario_id
            self.client.table("historical_financials").upsert(records).execute()
            return True
        except Exception as e:
            st.error(f"Error saving financials by user: {e}")
            return False

    def get_historical_customers(self, user_id: str) -> List[str]:
        try:
            response = (
                self.client.table("historical_financials")
                .select("customer_name")
                .eq("user_id", user_id)
                .execute()
            )
            if response.data:
                customers = sorted(list(set(row['customer_name'] for row in response.data if row.get('customer_name'))))
                return customers
            return []
        except Exception as e:
            st.error(f"Error loading customers: {e}")
            return []

    # =========================================================================
    # 4. LEGACY WEAR PROFILES & INSTALLED BASE (Fleet)
    # =========================================================================
    def get_wear_profiles(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Legacy method - returns dict keyed by machine_model."""
        try:
            response = self.client.table("wear_profiles").select("*").eq("user_id", user_id).execute()
            if response.data:
                profiles = {}
                for row in response.data:
                    model = row.get("machine_model")
                    if model:
                        profiles[model] = {
                            "liner_life_months": row.get("liner_life_months", 12),
                            "refurb_interval_months": row.get("refurb_interval_months", 36),
                            "avg_consumable_revenue": row.get("avg_consumable_revenue", 0.0),
                            "avg_refurb_revenue": row.get("avg_refurb_revenue", 0.0),
                            "gross_margin_liner": row.get("gross_margin_liner", 0.35),
                            "gross_margin_refurb": row.get("gross_margin_refurb", 0.28)
                        }
                return profiles
            return {}
        except Exception as e:
            st.error(f"Error loading wear profiles: {e}")
            return {}

    def get_wear_profiles_df(self, user_id: str) -> pd.DataFrame:
        """Legacy method - returns DataFrame."""
        try:
            response = self.client.table("wear_profiles").select("*").eq("user_id", user_id).execute()
            if response.data:
                data = []
                for row in response.data:
                    data.append({
                        "Model Name": row.get("machine_model"),
                        "Liner Life (Mo)": row.get("liner_life_months", 12),
                        "Refurb Interval (Mo)": row.get("refurb_interval_months", 36),
                        "Rev per Liner ($)": row.get("avg_consumable_revenue", 0.0),
                        "Rev per Refurb ($)": row.get("avg_refurb_revenue", 0.0)
                    })
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading wear profiles DF: {e}")
            return pd.DataFrame()

    def upsert_wear_profiles(self, user_id: str, df: pd.DataFrame) -> bool:
        """Legacy method - saves wear profiles from DataFrame."""
        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "user_id": user_id,
                    "machine_model": row["Model Name"],
                    "liner_life_months": int(row["Liner Life (Mo)"]),
                    "refurb_interval_months": int(row["Refurb Interval (Mo)"]),
                    "avg_consumable_revenue": float(row["Rev per Liner ($)"]),
                    "avg_refurb_revenue": float(row["Rev per Refurb ($)"])
                })
            self.client.table("wear_profiles").upsert(records, on_conflict="user_id,machine_model").execute() 
            return True
        except Exception as e:
            st.error(f"Error saving wear profiles: {e}")
            return False

    def upsert_wear_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Legacy method - saves single wear profile."""
        try:
            profile_data['user_id'] = user_id
            self.client.table("wear_profiles").upsert(profile_data).execute()
            return True
        except Exception as e:
            st.error(f"Error saving wear profile: {e}")
            return False

    def get_installed_base(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Legacy method - gets installed base records."""
        try:
            response = self.client.table("installed_base").select("*").eq("scenario_id", scenario_id).execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading fleet: {e}")
            return []

    def get_installed_base_df(self, scenario_id: str, user_id: str) -> pd.DataFrame:
        """Legacy method - gets installed base as DataFrame."""
        try:
            data = self.get_installed_base(scenario_id)
            if data:
                df_data = []
                for row in data:
                    df_data.append({
                        "Machine ID": row.get("machine_id", row.get("id")),
                        "Customer": row.get("customer_name"),
                        "Site": row.get("site_name", ""),
                        "Model": row.get("machine_model"),
                        "Commission Date": row.get("commission_date"),
                        "_id": row.get("id")
                    })
                return pd.DataFrame(df_data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading fleet DF: {e}")
            return pd.DataFrame()

    def upsert_installed_base(self, scenario_id: str, user_id: str, df: pd.DataFrame) -> bool:
        """Legacy method - saves installed base from DataFrame."""
        try:
            records = []
            for _, row in df.iterrows():
                record = {
                    "scenario_id": scenario_id,
                    "user_id": user_id,
                    "machine_id": str(row["Machine ID"]), 
                    "customer_name": row["Customer"],
                    "site_name": row.get("Site", ""),
                    "machine_model": row["Model"],
                    "commission_date": str(row["Commission Date"]),
                    "status": "Active"
                }
                if "_id" in row and pd.notna(row["_id"]):
                    record["id"] = row["_id"]
                records.append(record)
            
            self.client.table("installed_base").upsert(records, on_conflict="scenario_id,machine_id").execute()
            return True
        except Exception as e:
            st.error(f"Error saving fleet: {e}")
            return False

    def upsert_installed_base_machine(self, scenario_id: str, machine_data: Dict[str, Any]) -> bool:
        """Legacy method - saves single machine."""
        try:
            machine_data['scenario_id'] = scenario_id
            self.client.table("installed_base").upsert(machine_data).execute()
            return True
        except Exception as e:
            st.error(f"Error saving machine: {e}")
            return False

    # =========================================================================
    # 5. SUPPLY CHAIN (Suppliers & Trade Finance)
    # =========================================================================
    def get_suppliers(self, user_id: str) -> pd.DataFrame:
        try:
            response = self.client.table("suppliers").select("*").eq("user_id", user_id).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def upsert_supplier(self, user_id: str, supplier_data: Dict[str, Any]) -> bool:
        try:
            supplier_data['user_id'] = user_id
            self.client.table("suppliers").upsert(supplier_data).execute()
            return True
        except Exception as e:
            st.error(f"Error saving supplier: {e}")
            return False

    def get_trade_facilities(self, user_id: str) -> pd.DataFrame:
        try:
            response = self.client.table("trade_finance_facilities").select("*").eq("user_id", user_id).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def upsert_trade_facility(self, user_id: str, data: Dict[str, Any]) -> bool:
        try:
            data['user_id'] = user_id
            self.client.table("trade_finance_facilities").upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Error saving facility: {e}")
            return False

    # =========================================================================
    # 6. WORKING CAPITAL & GRANULAR DATA
    # =========================================================================
    def get_aged_debtors(self, user_id: str) -> pd.DataFrame:
        try:
            response = self.client.table("aged_debtors").select("*").eq("user_id", user_id).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading debtors: {e}")
            return pd.DataFrame()

    def get_aged_creditors(self, user_id: str) -> pd.DataFrame:
        try:
            response = self.client.table("aged_creditors").select("*").eq("user_id", user_id).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading creditors: {e}")
            return pd.DataFrame()

    def get_granular_sales(self, user_id: str) -> pd.DataFrame:
        try:
            response = self.client.table('granular_sales_history').select('*').eq('user_id', user_id).execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading granular sales: {e}")
            return pd.DataFrame()

    # =========================================================================
    # 7. REFERENCE DATA (Ore Types & Machine Models)
    # =========================================================================
    def get_ore_types(self) -> List[Dict[str, Any]]:
        """Get all ore types from reference table."""
        try:
            response = self.client.table("ore_types").select("*").order("name").execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading ore types: {e}")
            return []

    def get_machine_models(self) -> List[Dict[str, Any]]:
        """Get all machine models from reference table."""
        try:
            response = self.client.table("machine_models").select("*").order("code").execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading machine models: {e}")
            return []

    # =========================================================================
    # 8. CUSTOMER MANAGEMENT (Sprint 2)
    # =========================================================================
    def get_customers(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all customers for a user."""
        try:
            response = (
                self.client.table("customers")
                .select("*")
                .eq("user_id", user_id)
                .order("customer_name")
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading customers: {e}")
            return []

    def create_customer(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new customer."""
        try:
            data['user_id'] = user_id
            result = self.client.table("customers").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error creating customer: {e}")
            raise e

    def update_customer(self, customer_id: str, user_id: str, data: Dict[str, Any]) -> bool:
        """Update an existing customer."""
        try:
            self.client.table("customers").update(data).eq("id", customer_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating customer: {e}")
            raise e

    def delete_customer(self, customer_id: str, user_id: str) -> bool:
        """Delete a customer."""
        try:
            self.client.table("customers").delete().eq("id", customer_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting customer: {e}")
            raise e

    # =========================================================================
    # 9. SITE MANAGEMENT (Sprint 2)
    # =========================================================================
    def get_sites(self, user_id: str, customer_id: str = None) -> List[Dict[str, Any]]:
        """Get sites, optionally filtered by customer."""
        try:
            query = (
                self.client.table("sites")
                .select("*, customers(customer_code, customer_name), ore_types(name, code)")
                .eq("user_id", user_id)
            )
            if customer_id:
                query = query.eq("customer_id", customer_id)
            response = query.order("site_name").execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading sites: {e}")
            return []

    def create_site(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new site."""
        try:
            data['user_id'] = user_id
            result = self.client.table("sites").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error creating site: {e}")
            raise e

    def update_site(self, site_id: str, user_id: str, data: Dict[str, Any]) -> bool:
        """Update an existing site."""
        try:
            self.client.table("sites").update(data).eq("id", site_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating site: {e}")
            raise e

    def delete_site(self, site_id: str, user_id: str) -> bool:
        """Delete a site."""
        try:
            self.client.table("sites").delete().eq("id", site_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting site: {e}")
            raise e

    # =========================================================================
    # 10. MACHINE INSTANCE MANAGEMENT (Sprint 2)
    # =========================================================================
    def get_machine_instances(self, user_id: str, scenario_id: str, site_id: str = None) -> List[Dict[str, Any]]:
        """Get machine instances with full hierarchy."""
        try:
            query = (
                self.client.table("machine_instances")
                .select("*, sites(site_name, ore_type_id, customers(customer_code, customer_name)), machine_models(code, name)")
                .eq("user_id", user_id)
                .eq("scenario_id", scenario_id)
            )
            if site_id:
                query = query.eq("site_id", site_id)
            response = query.execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading machine instances: {e}")
            return []

    def get_machine_instances_df(self, user_id: str, scenario_id: str) -> pd.DataFrame:
        """Get machine instances as DataFrame for forecast engine."""
        try:
            data = self.get_machine_instances(user_id, scenario_id)
            if data:
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading machine instances DF: {e}")
            return pd.DataFrame()

    def create_machine_instance(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new machine instance."""
        try:
            data['user_id'] = user_id
            result = self.client.table("machine_instances").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error creating machine instance: {e}")
            raise e

    def update_machine_instance(self, machine_id: str, user_id: str, data: Dict[str, Any]) -> bool:
        """Update an existing machine instance."""
        try:
            self.client.table("machine_instances").update(data).eq("id", machine_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating machine instance: {e}")
            raise e

    def delete_machine_instance(self, machine_id: str, user_id: str) -> bool:
        """Delete a machine instance."""
        try:
            self.client.table("machine_instances").delete().eq("id", machine_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting machine instance: {e}")
            raise e

    # =========================================================================
    # 11. WEAR PROFILES V2 - Ore Type Aware (Sprint 4)
    # =========================================================================
    def get_wear_profiles_v2(self, user_id: str) -> List[Dict[str, Any]]:
        """Get wear profiles with ore type and model relationships."""
        try:
            response = (
                self.client.table("wear_profiles")
                .select("*, ore_types(name, code), machine_models(code, name)")
                .eq("user_id", user_id)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading wear profiles v2: {e}")
            return []

    def upsert_wear_profile_v2(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Upsert wear profile with ore type support."""
        try:
            data['user_id'] = user_id
            self.client.table("wear_profiles").upsert(
                data, 
                on_conflict="user_id,machine_model_id,ore_type_id"
            ).execute()
            return True
        except Exception as e:
            st.error(f"Error saving wear profile v2: {e}")
            raise e

    # =========================================================================
    # 12. PROSPECT/PIPELINE MANAGEMENT (Sprint 5)
    # =========================================================================
    def get_prospects(self, user_id: str, scenario_id: str = None, stage: str = None) -> List[Dict[str, Any]]:
        """Get prospects with optional filtering."""
        try:
            query = (
                self.client.table("prospects")
                .select("*, ore_types(name, code), machine_models(code, name)")
                .eq("user_id", user_id)
            )
            if scenario_id:
                query = query.eq("scenario_id", scenario_id)
            if stage:
                query = query.eq("pipeline_stage", stage)
            response = query.order("expected_close_date").execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading prospects: {e}")
            return []

    def create_prospect(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new prospect."""
        try:
            data['user_id'] = user_id
            result = self.client.table("prospects").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error creating prospect: {e}")
            raise e

    def update_prospect(self, prospect_id: str, user_id: str, data: Dict[str, Any]) -> bool:
        """Update an existing prospect."""
        try:
            self.client.table("prospects").update(data).eq("id", prospect_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating prospect: {e}")
            raise e

    def delete_prospect(self, prospect_id: str, user_id: str) -> bool:
        """Delete a prospect."""
        try:
            self.client.table("prospects").delete().eq("id", prospect_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting prospect: {e}")
            raise e

    # =========================================================================
    # 13. FORECAST SNAPSHOTS (Sprint 6)
    # =========================================================================
    def save_forecast_snapshot(self, user_id: str, scenario_id: str, snapshot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save a forecast snapshot for comparison.
        
        Args:
            user_id: User ID
            scenario_id: Scenario ID
            snapshot_data: Dict containing snapshot_name, snapshot_date, snapshot_type,
                          assumptions_data, forecast_data, prospects_data, valuation_data,
                          total_revenue_forecast, notes, etc.
        
        Returns:
            Created snapshot record or None
        """
        try:
            data = {
                'user_id': user_id,
                'scenario_id': scenario_id,
                **snapshot_data
            }
            result = self.client.table("forecast_snapshots").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error saving snapshot: {e}")
            raise e

    def get_forecast_snapshots(self, user_id: str, scenario_id: str = None) -> List[Dict[str, Any]]:
        """Get forecast snapshots, optionally filtered by scenario."""
        try:
            query = self.client.table("forecast_snapshots").select("*").eq("user_id", user_id)
            if scenario_id:
                query = query.eq("scenario_id", scenario_id)
            response = query.order("created_at", desc=True).execute()
            return response.data if response.data else []
        except Exception as e:
            st.error(f"Error loading snapshots: {e}")
            return []

    def get_forecast_snapshot_by_id(self, snapshot_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific snapshot by ID."""
        try:
            response = (
                self.client.table("forecast_snapshots")
                .select("*")
                .eq("id", snapshot_id)
                .eq("user_id", user_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            st.error(f"Error loading snapshot: {e}")
            return None

    def update_forecast_snapshot(self, snapshot_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a forecast snapshot (notes, is_locked, snapshot_name only).
        """
        try:
            # Only allow certain fields to be updated
            allowed_fields = ['notes', 'is_locked', 'snapshot_name']
            safe_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not safe_updates:
                return None
            
            result = (
                self.client.table("forecast_snapshots")
                .update(safe_updates)
                .eq("id", snapshot_id)
                .eq("user_id", user_id)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error updating snapshot: {e}")
            raise e

    def delete_forecast_snapshot(self, snapshot_id: str, user_id: str) -> bool:
        """Delete a forecast snapshot (if not locked)."""
        try:
            # Check if snapshot is locked
            snapshot = self.get_forecast_snapshot_by_id(snapshot_id, user_id)
            if snapshot and snapshot.get('is_locked'):
                st.error("Cannot delete a locked snapshot. Unlock it first.")
                return False
            
            self.client.table("forecast_snapshots").delete().eq("id", snapshot_id).eq("user_id", user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting snapshot: {e}")
            return False

    def lock_forecast_snapshot(self, snapshot_id: str, user_id: str) -> bool:
        """Lock a snapshot to prevent deletion."""
        return self.update_forecast_snapshot(snapshot_id, user_id, {'is_locked': True}) is not None

    def unlock_forecast_snapshot(self, snapshot_id: str, user_id: str) -> bool:
        """Unlock a snapshot to allow deletion."""
        return self.update_forecast_snapshot(snapshot_id, user_id, {'is_locked': False}) is not None

    # =========================================================================
    # 14. HISTORIC DATA TABLES (for imports)
    # =========================================================================
    def get_historic_customers(self, scenario_id: str) -> List[Dict[str, Any]]:
        """Get historic customers for a scenario."""
        try:
            response = (
                self.client.table("historic_customers")
                .select("*")
                .eq("scenario_id", scenario_id)
                .order("customer_name")
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            print(f"Error loading historic customers: {e}")
            return []

    # =========================================================================
    # 15. EXPENSE ASSUMPTIONS (Sprint 7 - NEW)
    # =========================================================================
    def get_expense_assumptions(self, scenario_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all expense assumptions for a scenario.
        
        Args:
            scenario_id: Scenario UUID
            active_only: If True, only return active assumptions
        
        Returns:
            List of expense assumption records
        """
        try:
            query = self.client.table('expense_assumptions').select('*').eq('scenario_id', scenario_id)
            
            if active_only:
                query = query.eq('is_active', True)
            
            result = query.order('expense_category').order('expense_name').execute()
            return result.data or []
        except Exception as e:
            print(f"Error fetching expense assumptions: {e}")
            return []

    def get_expense_assumption(self, scenario_id: str, expense_code: str) -> Optional[Dict[str, Any]]:
        """Get a single expense assumption by code."""
        try:
            result = (
                self.client.table('expense_assumptions')
                .select('*')
                .eq('scenario_id', scenario_id)
                .eq('expense_code', expense_code)
                .single()
                .execute()
            )
            return result.data
        except Exception as e:
            print(f"Error fetching expense assumption: {e}")
            return None

    def upsert_expense_assumption(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert or update an expense assumption.
        
        Args:
            data: Dict with expense assumption fields (must include scenario_id, expense_code)
        
        Returns:
            Upserted record or None
        """
        try:
            # Ensure required fields
            if 'scenario_id' not in data or 'expense_code' not in data:
                raise ValueError("scenario_id and expense_code are required")
            
            # Add updated_at timestamp
            data['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table('expense_assumptions').upsert(
                data,
                on_conflict='scenario_id,expense_code'
            ).execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error upserting expense assumption: {e}")
            return None

    def delete_expense_assumption(self, scenario_id: str, expense_code: str) -> bool:
        """Delete an expense assumption."""
        try:
            self.client.table('expense_assumptions').delete().eq('scenario_id', scenario_id).eq('expense_code', expense_code).execute()
            return True
        except Exception as e:
            print(f"Error deleting expense assumption: {e}")
            return False

    def bulk_upsert_expense_assumptions(self, scenario_id: str, assumptions: List[Dict[str, Any]]) -> int:
        """
        Bulk upsert multiple expense assumptions.
        
        Args:
            scenario_id: Target scenario
            assumptions: List of assumption dicts
        
        Returns:
            Number of records upserted
        """
        try:
            now = datetime.now().isoformat()
            
            # Add scenario_id and timestamp to all records
            for a in assumptions:
                a['scenario_id'] = scenario_id
                a['updated_at'] = now
            
            result = self.client.table('expense_assumptions').upsert(
                assumptions,
                on_conflict='scenario_id,expense_code'
            ).execute()
            
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"Error bulk upserting expense assumptions: {e}")
            return 0

    def get_expense_assumptions_by_category(self, scenario_id: str, category: str) -> List[Dict[str, Any]]:
        """Get expense assumptions for a specific category."""
        try:
            result = (
                self.client.table('expense_assumptions')
                .select('*')
                .eq('scenario_id', scenario_id)
                .eq('expense_category', category)
                .eq('is_active', True)
                .execute()
            )
            return result.data or []
        except Exception as e:
            print(f"Error fetching expense assumptions by category: {e}")
            return []

    def get_expense_correlation_groups(self, scenario_id: str) -> Dict[str, List[str]]:
        """
        Get expense assumptions organized by correlation group.
        
        Returns:
            Dict of {group_name: [expense_codes]}
        """
        try:
            assumptions = self.get_expense_assumptions(scenario_id)
            
            groups = {}
            for a in assumptions:
                group = a.get('correlation_group')
                if group:
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(a['expense_code'])
            
            return groups
        except Exception as e:
            print(f"Error fetching correlation groups: {e}")
            return {}

    def calculate_total_opex(self, scenario_id: str, monthly_revenue: float, month_number: int = 0) -> Dict[str, float]:
        """
        Calculate total OPEX for a given revenue level using stored assumptions.
        
        Args:
            scenario_id: Scenario UUID
            monthly_revenue: Revenue for the month
            month_number: Months from start (for escalation)
        
        Returns:
            Dict with {expense_code: amount, ..., 'total': sum}
        """
        try:
            assumptions = self.get_expense_assumptions(scenario_id)
            
            result = {}
            total = 0.0
            
            for a in assumptions:
                code = a['expense_code']
                func_type = a.get('function_type', 'fixed')
                
                if func_type == 'fixed':
                    # Fixed monthly amount
                    amount = float(a.get('fixed_monthly', 0) or 0)
                    
                elif func_type == 'variable':
                    # Percentage of revenue
                    rate = float(a.get('variable_rate', 0) or 0)
                    amount = monthly_revenue * rate
                    
                elif func_type == 'stepped':
                    # Fixed amount that changes at certain thresholds
                    amount = float(a.get('fixed_monthly', 0) or 0)
                    
                elif func_type == 'interpolation':
                    # Linear interpolation from start to end over forecast period
                    start_val = float(a.get('start_value', 0) or 0)
                    end_val = float(a.get('end_value', 0) or 0)
                    # Assume 60-month forecast for interpolation
                    if month_number >= 60:
                        amount = end_val
                    else:
                        progress = month_number / 60
                        amount = start_val + (end_val - start_val) * progress
                else:
                    amount = 0.0
                
                result[code] = amount
                total += amount
            
            result['total'] = total
            return result
            
        except Exception as e:
            print(f"Error calculating total opex: {e}")
            return {'total': 0.0}

    def clone_expense_assumptions(self, source_scenario_id: str, target_scenario_id: str) -> int:
        """
        Clone expense assumptions from one scenario to another.
        
        Args:
            source_scenario_id: Source scenario UUID
            target_scenario_id: Target scenario UUID
        
        Returns:
            Number of records cloned
        """
        try:
            # Get source assumptions
            source_assumptions = self.get_expense_assumptions(source_scenario_id, active_only=False)
            
            if not source_assumptions:
                return 0
            
            # Prepare for target
            now = datetime.now().isoformat()
            
            target_assumptions = []
            for a in source_assumptions:
                new_record = {k: v for k, v in a.items() if k not in ['id', 'created_at', 'updated_at']}
                new_record['scenario_id'] = target_scenario_id
                new_record['updated_at'] = now
                target_assumptions.append(new_record)
            
            # Insert into target
            result = self.client.table('expense_assumptions').upsert(
                target_assumptions,
                on_conflict='scenario_id,expense_code'
            ).execute()
            
            return len(result.data) if result.data else 0
            
        except Exception as e:
            print(f"Error cloning expense assumptions: {e}")
            return 0
